"""Worker de LiveKit Agents: marca al cliente por la troncal SIP de Zadarma y sigue el guion.

Flujo de cada llamada (un "job" por sala):
1. El dispatcher crea la sala y despacha este agente con la venta en `metadata`.
2. El agente arranca la sesión de voz y marca al celular del cliente (SIP, espera a que conteste).
3. Al contestar suena un pitido y el agente saluda de una: no espera ningún "hola" del cliente.
4. El AMD (detección de buzón) corre en paralelo mientras saluda: si es un buzón, corta y cuelga.
5. Al colgar, copia la grabación y escribe result.json; el dispatcher entrega a n8n.

En una simulación (metadata "simulacion": true) no se marca: el "cliente" entra a la sala desde
el navegador (la página del dispatcher) y el resto del flujo es idéntico.

Ejecutar:  python -m voice_confirm.agent start
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

import httpx
from google.protobuf.duration_pb2 import Duration
from livekit import api, rtc
from livekit.agents import (
    AMD,
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    JobContext,
    JobProcess,
    ModelSettings,
    RunContext,
    ToolResult,
    cli,
    function_tool,
    get_job_context,
    metrics,
    room_io,
)
from livekit.plugins import assemblyai, silero

from .config import AgentModels, Settings
from .domain import (
    CEDULA_REPREGUNTA,
    CEDULA_RETRY,
    CLIENT_IDENTITY,
    GOODBYE_DATOS,
    GOODBYE_OK,
    GOODBYE_REVIEW,
    CallStatus,
    Sale,
    acceptance_prompt,
    build_instructions,
    digits_only,
    evaluate,
    greeting_text,
    parece_ingles,
    phone_repregunta,
    purchase_text,
    status_from_sip_code,
)
from .outbox import call_dir, utcnow, write_json_atomic

logger = logging.getLogger("genix.confirmacion")

_trunk_cache: dict[str, str] = {}
SIM_JOIN_TIMEOUT_S = 120  # tiempo para que el navegador entre a la sala
SIM_MIC_TIMEOUT_S = 10  # tiempo para que el navegador publique el micrófono
SILENCIO_REPREGUNTA_S = 8.0  # silencio del cliente tras una pregunta antes de repetirla
MAX_REPREGUNTAS = 2
SALUDO_ESPERA_S = 3.0  # si el saludo pre-sintetizado no está listo a tiempo, se dice con TTS en vivo
PITIDO_DURACION_S = 0.35  # lo que dura el pitido de inicio: se deja sonar antes de saludar
AMD_VEREDICTO_S = 7.0  # el AMD corre mientras el agente saluda; pasado esto se asume persona


@dataclass
class CallState:
    cedula_intentos: int = 0
    cedula_dicha: str | None = None
    cedula_coincide: bool | None = None
    celular_confirmado: bool | None = None
    dijo_acepto: bool | None = None
    frase_aceptacion: str | None = None
    observacion: str | None = None
    motivo_fin: str | None = None
    status_override: CallStatus | None = None
    contestada: datetime | None = None
    simulacion: bool = False
    pregunta: str = CEDULA_REPREGUNTA  # lo que se repite si el cliente se queda callado
    repreguntas: int = 0
    terminando: bool = False
    metricas: list[dict[str, Any]] = field(default_factory=list)


async def hangup() -> None:
    ctx = get_job_context()
    await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))


PITIDO_VOLUMEN = 0.35
_PITIDO_RATE = 48000
_PITIDO_FRAME = _PITIDO_RATE // 100  # 10 ms


def _pitido_pcm() -> bytes:
    """Dos tonos cortos (880 Hz y 1320 Hz) con rampa suave: suena a "recibido", no a error."""
    muestras: list[int] = []
    for freq, ms in ((880, 90), (0, 40), (1320, 110)):
        n = _PITIDO_RATE * ms // 1000
        rampa = _PITIDO_RATE * 8 // 1000
        for i in range(n):
            env = min(1.0, i / rampa, (n - i) / rampa) if freq else 0.0
            muestras.append(int(32767 * 0.6 * env * math.sin(2 * math.pi * freq * i / _PITIDO_RATE)))
    muestras += [0] * (-len(muestras) % _PITIDO_FRAME)
    return b"".join(m.to_bytes(2, "little", signed=True) for m in muestras)


_PITIDO = _pitido_pcm()


async def _pitido_frames() -> AsyncIterator[rtc.AudioFrame]:
    paso = _PITIDO_FRAME * 2
    for i in range(0, len(_PITIDO), paso):
        yield rtc.AudioFrame(_PITIDO[i : i + paso], _PITIDO_RATE, 1, _PITIDO_FRAME)


class AudioListo:
    """Sintetiza de antemano los textos fijos: suenan de inmediato y en un solo bloque, sin esperar al TTS."""

    def __init__(self, tts: Any) -> None:
        self._tts = tts
        self._tareas: dict[str, asyncio.Task[list[rtc.AudioFrame]]] = {}

    def preparar(self, *textos: str) -> None:
        for texto in textos:
            if texto not in self._tareas:
                self._tareas[texto] = asyncio.create_task(self._sintetizar(texto))

    async def _sintetizar(self, texto: str) -> list[rtc.AudioFrame]:
        frames: list[rtc.AudioFrame] = []
        async with self._tts.synthesize(texto) as stream:
            async for ev in stream:
                frames.append(ev.frame)
        return frames

    async def esperar(self, texto: str, timeout: float) -> list[rtc.AudioFrame] | None:
        tarea = self._tareas.get(texto)
        if tarea is None:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(tarea), timeout)
        except asyncio.TimeoutError:
            return None
        except Exception:
            logger.warning("no se pudo pre-sintetizar: %r", texto[:60], exc_info=True)
            return None

    def listo(self, texto: str) -> list[rtc.AudioFrame] | None:
        tarea = self._tareas.get(texto)
        if tarea is None or not tarea.done() or tarea.cancelled() or tarea.exception() is not None:
            return None
        return tarea.result()

    def cancelar(self) -> None:
        for tarea in self._tareas.values():
            tarea.cancel()


async def _reproducir(frames: list[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
    for frame in frames:
        yield frame


def decir(session: AgentSession, texto: str, audio: AudioListo | None, **kwargs: Any) -> Any:
    """session.say con el audio ya sintetizado si está listo; si no, TTS en vivo como siempre."""
    frames = audio.listo(texto) if audio is not None else None
    if frames:
        return session.say(texto, audio=_reproducir(frames), **kwargs)
    return session.say(texto, **kwargs)


_FRASE = re.compile(r"[^.!?\n]*[.!?\n]+")


async def _solo_espanol(text: AsyncIterable[str]) -> AsyncIterator[str]:
    """Deja pasar al TTS frase por frase y descarta las que el modelo escribió en inglés."""
    pendiente = ""
    async for trozo in text:
        pendiente += trozo
        corte = 0
        for m in _FRASE.finditer(pendiente):
            corte = m.end()
            if parece_ingles(m.group()):
                logger.warning("frase descartada antes del TTS (no es español): %r", m.group()[:120])
            else:
                yield m.group()
        pendiente = pendiente[corte:]
    if pendiente.strip():
        if parece_ingles(pendiente):
            logger.warning("frase descartada antes del TTS (no es español): %r", pendiente[:120])
        else:
            yield pendiente


class ConfirmacionAgent(Agent):
    def __init__(self, sale: Sale, state: CallState) -> None:
        super().__init__(instructions=build_instructions(sale))
        self._sale = sale
        self._state = state
        self.fondo: BackgroundAudioPlayer | None = None
        self.audio: AudioListo | None = None
        self._tareas: set[asyncio.Task[None]] = set()

    # Las herramientas dicen el texto fijo ellas mismas y piden al LLM no responder:
    # así cada paso cuesta una sola llamada al modelo, no dos.
    def _decir(self, context: RunContext, texto: str, pregunta: str | None = None) -> ToolResult:
        if pregunta is not None:
            self._state.pregunta = pregunta
            self._state.repreguntas = 0
        decir(context.session, texto, self.audio)
        return ToolResult("Ya se le dijo al cliente. No digas nada; espera su respuesta.", reply_required=False)

    def despedir(self, session: AgentSession, texto: str, motivo: str) -> None:
        """Dice la despedida y cuelga cuando termina de sonar."""
        s = self._state
        if s.terminando:
            return
        s.terminando = True
        s.motivo_fin = s.motivo_fin or motivo
        handle = decir(session, texto, self.audio, allow_interruptions=False)

        async def colgar() -> None:
            await handle
            await hangup()

        # No se espera aquí: la herramienta en curso es parte de la cola de voz y se bloquearía.
        tarea = asyncio.create_task(colgar())
        self._tareas.add(tarea)
        tarea.add_done_callback(self._tareas.discard)

    def pitido(self) -> None:
        """Suena el tono corto. Se usa al abrir la llamada y cada vez que el cliente termina de hablar."""
        if self.fondo is None:
            return
        try:
            self.fondo.play(AudioConfig(_pitido_frames(), volume=PITIDO_VOLUMEN))
        except Exception:
            logger.warning("no se pudo reproducir el pitido", exc_info=True)

    async def on_user_turn_completed(self, turn_ctx: Any, new_message: Any) -> None:
        # Pitido apenas el cliente termina de hablar: sabe que se le escuchó mientras se valida.
        self.pitido()

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        async for frame in Agent.default.tts_node(self, _solo_espanol(text), model_settings):
            yield frame

    def _fin(self, context: RunContext, texto: str, motivo: str) -> ToolResult:
        self.despedir(context.session, texto, motivo)
        return ToolResult("Despedida en curso; la llamada se cuelga sola. No digas nada.", reply_required=False)

    @function_tool
    async def registrar_cedula(self, context: RunContext, cedula: str) -> ToolResult:
        """Registra la cédula que dijo el cliente.

        Args:
            cedula: Solo los dígitos de la cédula que dijo el cliente, sin puntos ni espacios.
        """
        s = self._state
        s.cedula_intentos += 1
        dicha = digits_only(cedula)
        s.cedula_dicha = dicha
        if dicha and dicha == self._sale.cedula:
            s.cedula_coincide = True
            return self._decir(context, purchase_text(self._sale), phone_repregunta(self._sale))
        if s.cedula_intentos < 2:
            return self._decir(context, CEDULA_RETRY, CEDULA_REPREGUNTA)
        s.cedula_coincide = False
        return self._fin(context, GOODBYE_REVIEW, "cedula_no_coincide")

    # El agente solo valida lo que está en el formulario: nunca recibe ni guarda datos nuevos.
    # El correo no pasa por aquí: se confirma en el formulario.
    @function_tool
    async def registrar_celular(self, context: RunContext, correcto: bool) -> ToolResult:
        """Registra si el cliente confirmó que el celular del formulario es el suyo.

        Args:
            correcto: true si dijo que sí es su número; false si dijo que no, que está mal o pidió cambiarlo.
        """
        self._state.celular_confirmado = correcto
        if correcto:
            texto = acceptance_prompt(self._sale)
            return self._decir(context, texto, texto)
        self._state.observacion = "El cliente indica que el celular del formulario no es correcto."
        return self._fin(context, GOODBYE_DATOS, "celular_incorrecto")

    @function_tool
    async def registrar_aceptacion(self, context: RunContext, acepta: bool, frase: str) -> ToolResult:
        """Registra si el cliente dijo en voz alta que acepta la compra.

        Args:
            acepta: true solo si dijo que acepta la compra.
            frase: Lo que dijo el cliente, tal como lo dijo.
        """
        self._state.dijo_acepto = acepta
        self._state.frase_aceptacion = frase[:300]
        if acepta:
            return self._fin(context, GOODBYE_OK, "completada")
        return self._fin(context, GOODBYE_REVIEW, "no_acepta")

    @function_tool
    async def terminar_llamada(self, context: RunContext, motivo: str, observacion: str = "") -> ToolResult | None:
        """Se despide y cuelga cuando el cliente se sale del guion; con motivo "buzon" cuelga sin hablar.

        Args:
            motivo: completada, buzon, cliente_no_continua, cedula_no_coincide, no_confirma, no_acepta,
                celular_incorrecto, datos_incorrectos, pide_asesor o pregunta_fuera_de_guion.
            observacion: Si no terminó confirmada, resumen corto en español de lo que pidió el cliente,
                sin copiar datos nuevos que haya dictado (correos, números, cédulas).
        """
        if observacion and not self._state.observacion:
            self._state.observacion = observacion[:300]
        if self._state.terminando:  # ya se está despidiendo una herramienta anterior
            return ToolResult("La llamada ya se está cerrando. No digas nada.", reply_required=False)
        if motivo == "buzon":
            self._state.motivo_fin = motivo
            self._state.status_override = "buzon"
            await context.wait_for_playout()
            await hangup()
            return None
        texto = GOODBYE_DATOS if motivo == "datos_incorrectos" else GOODBYE_REVIEW
        return self._fin(context, texto, motivo)

    def repreguntar(self, session: AgentSession) -> None:
        """El cliente lleva SILENCIO_REPREGUNTA_S callado: repite la pregunta corta, o se despide."""
        s = self._state
        if s.terminando:
            return
        if s.repreguntas >= MAX_REPREGUNTAS:
            s.observacion = s.observacion or "El cliente dejó de responder durante la llamada."
            self.despedir(session, GOODBYE_REVIEW, "cliente_no_responde")
            return
        s.repreguntas += 1
        logger.info("repregunta %d por silencio del cliente", s.repreguntas)
        session.say(s.pregunta)


def build_stt(m: AgentModels) -> assemblyai.STT:
    kwargs: dict[str, Any] = {}
    if m.stt_max_turn_silence_ms:
        kwargs["max_turn_silence"] = m.stt_max_turn_silence_ms
    if not m.stt_model.startswith("universal-streaming"):
        kwargs["language_codes"] = ["es"]
    return assemblyai.STT(api_key=m.assemblyai_api_key, model=m.stt_model, **kwargs)


def build_llm(m: AgentModels) -> Any:
    if m.llm_provider == "anthropic":
        from livekit.plugins import anthropic

        return anthropic.LLM(model=m.llm_model, api_key=m.anthropic_api_key)

    from livekit.plugins import openai

    kwargs: dict[str, Any] = {}
    if m.llm_reasoning == "off":
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    elif m.llm_reasoning == "low":
        kwargs["extra_body"] = {"reasoning": {"effort": "low"}}
    return openai.LLM(
        model=m.llm_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=m.openrouter_api_key,
        **kwargs,
    )


def build_tts(m: AgentModels) -> Any:
    if m.tts_provider == "elevenlabs":
        from livekit.plugins import elevenlabs

        return elevenlabs.TTS(
            api_key=m.elevenlabs_api_key,
            voice_id=m.elevenlabs_voice_id,
            model=m.elevenlabs_model,
            language="es",
        )
    from livekit.plugins import fishaudio

    return fishaudio.TTS(api_key=m.fish_api_key, model=m.fish_model, voice_id=m.fish_voice_id)


async def resolve_trunk_id(lkapi: api.LiveKitAPI, trunk_name: str) -> str:
    if trunk_name in _trunk_cache:
        return _trunk_cache[trunk_name]
    resp = await lkapi.sip.list_outbound_trunk(api.ListSIPOutboundTrunkRequest())
    for trunk in resp.items:
        if trunk.name == trunk_name:
            _trunk_cache[trunk_name] = trunk.sip_trunk_id
            return trunk.sip_trunk_id
    raise RuntimeError(
        f"No existe la troncal SIP '{trunk_name}'. Ejecute: docker compose run --rm dispatcher python -m voice_confirm.trunk"
    )


def transcript(session: AgentSession | None) -> list[dict[str, str]]:
    if session is None:
        return []
    lines: list[dict[str, str]] = []
    for item in session.history.items:
        if getattr(item, "type", None) != "message" or item.role not in ("user", "assistant"):
            continue
        text = (item.text_content or "").strip()
        if text:
            lines.append({"rol": "cliente" if item.role == "user" else "agente", "texto": text})
    return lines


async def wait_for_microphone(room: rtc.Room, identity: str, timeout: float) -> None:
    """Espera a que el navegador publique su micrófono: así el saludo no se pierde mientras conecta."""
    participant = room.remote_participants.get(identity)
    if participant and any(p.kind == rtc.TrackKind.KIND_AUDIO for p in participant.track_publications.values()):
        return
    published: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def on_published(publication: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant) -> None:
        if p.identity == identity and publication.kind == rtc.TrackKind.KIND_AUDIO and not published.done():
            published.set_result(None)

    room.on("track_published", on_published)
    try:
        await asyncio.wait_for(published, timeout)
    except asyncio.TimeoutError:
        logger.warning("el navegador no publicó el micrófono a tiempo; se saluda igual")
    finally:
        room.off("track_published", on_published)


async def answer_simulation(ctx: JobContext, settings: Settings, state: CallState) -> bool:
    """Simulación: en vez de marcar, espera al "cliente" del navegador. False si nunca entró."""
    try:
        await asyncio.wait_for(ctx.wait_for_participant(identity=CLIENT_IDENTITY), SIM_JOIN_TIMEOUT_S)
    except asyncio.TimeoutError:
        state.status_override = "no_contesta"
        state.motivo_fin = "simulacion_sin_participante"
        ctx.shutdown(reason=state.motivo_fin)
        return False
    await wait_for_microphone(ctx.room, CLIENT_IDENTITY, SIM_MIC_TIMEOUT_S)
    await asyncio.sleep(0.5)  # margen para que el navegador termine de suscribirse al audio del agente
    state.contestada = utcnow()

    # En una llamada real el límite lo pone max_call_duration del SIP; aquí se cuelga a mano.
    async def limit_duration() -> None:
        await asyncio.sleep(settings.max_call_duration_s)
        state.motivo_fin = state.motivo_fin or "duracion_maxima"
        await hangup()

    limiter = asyncio.create_task(limit_duration())

    async def cancel_limiter() -> None:
        limiter.cancel()

    ctx.add_shutdown_callback(cancel_limiter)
    return True


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=prewarm, shutdown_process_timeout=30.0)
_agent_name = Settings.agent_name_from_env()


@server.rtc_session(agent_name=_agent_name)
async def entrypoint(ctx: JobContext) -> None:
    settings = Settings.load()
    models = AgentModels.load()
    payload = json.loads(ctx.job.metadata or "{}")
    sale = Sale.from_dict(payload["sale"])
    call_id = ctx.room.name
    state = CallState(simulacion=bool(payload.get("simulacion", False)))
    holder: dict[str, AgentSession] = {}

    async def on_shutdown(reason: str) -> None:
        await finalize(ctx, settings, call_id, sale, state, holder.get("session"), reason)

    ctx.add_shutdown_callback(on_shutdown)
    await ctx.connect()

    trunk_id = ""
    if not state.simulacion:
        try:
            trunk_id = await resolve_trunk_id(ctx.api, settings.trunk_name)
        except Exception:
            logger.exception("no se encontró la troncal SIP")
            state.status_override = "fallida"
            state.motivo_fin = "sin_troncal_sip"
            ctx.shutdown(reason="sin_troncal_sip")
            return

    tts = build_tts(models)
    # Mientras timbra (o el navegador conecta) ya se sintetizan el saludo y los textos del camino feliz.
    audio = AudioListo(tts)
    audio.preparar(greeting_text(), purchase_text(sale), acceptance_prompt(sale), GOODBYE_OK)

    async def cancelar_audio() -> None:
        audio.cancelar()

    ctx.add_shutdown_callback(cancelar_audio)

    session: AgentSession = AgentSession(
        stt=build_stt(models),
        llm=build_llm(models),
        tts=tts,
        vad=ctx.proc.userdata["vad"],
        # El VAD cierra el turno a los 0,5 s de silencio: con "stt", AssemblyAI se tragaba los "sí" cortos.
        turn_handling={"turn_detection": "vad", "endpointing": {"min_delay": 0.5, "max_delay": 2.0}},
        user_away_timeout=SILENCIO_REPREGUNTA_S,
    )
    holder["session"] = session
    agent = ConfirmacionAgent(sale, state)
    agent.audio = audio

    async def saludar() -> None:
        # Primero el pitido: la persona oye que el proceso arrancó aunque no haya dicho nada.
        agent.pitido()
        await asyncio.sleep(PITIDO_DURACION_S)
        await audio.esperar(greeting_text(), SALUDO_ESPERA_S)
        decir(session, greeting_text(), audio, allow_interruptions=False)

    @session.on("metrics_collected")
    def on_metrics(ev: Any) -> None:
        m = ev.metrics
        if isinstance(m, metrics.EOUMetrics):
            fila = {"tipo": "fin_turno", "eou_s": round(m.end_of_utterance_delay, 3),
                    "transcripcion_s": round(m.transcription_delay, 3)}
        elif isinstance(m, metrics.LLMMetrics):
            fila = {"tipo": "llm", "ttft_s": round(m.ttft, 3)}
        elif isinstance(m, metrics.TTSMetrics):
            fila = {"tipo": "tts", "ttfb_s": round(m.ttfb, 3)}
        else:
            return
        state.metricas.append(fila)
        logger.info("latencia %s", fila, extra={"call_id": call_id, **fila})

    @session.on("user_state_changed")
    def on_user_state(ev: Any) -> None:
        if ev.new_state == "away" and state.contestada is not None:
            agent.repreguntar(session)
    await session.start(
        agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            participant_identity=CLIENT_IDENTITY,
            close_on_disconnect=True,
            delete_room_on_close=True,
        ),
        # Graba cliente + agente en un solo audio.ogg dentro de la carpeta temporal del job.
        # Sin LiveKit Cloud no se sube a ningún lado: finalize() lo copia a DATA_DIR.
        record={"audio": True, "traces": False, "logs": False, "transcript": False},
    )
    try:
        fondo = BackgroundAudioPlayer()
        await fondo.start(room=ctx.room, agent_session=session)
        agent.fondo = fondo

        async def close_fondo() -> None:
            await fondo.aclose()

        ctx.add_shutdown_callback(close_fondo)
    except Exception:
        logger.warning("sin audio de fondo: la llamada sigue sin pitido", exc_info=True)

    if state.simulacion:
        if await answer_simulation(ctx, settings, state):
            await saludar()
        return

    try:
        async with AMD(
            session,
            llm=None,  # reutiliza el LLM de la sesión
            stt=None,  # reutiliza las transcripciones de la sesión
            participant_identity=CLIENT_IDENTITY,
            ivr_detection=False,
            suppress_compatibility_warning=True,
        ) as detector:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=call_id,
                    sip_trunk_id=trunk_id,
                    sip_call_to=sale.telefono,
                    participant_identity=CLIENT_IDENTITY,
                    participant_name=sale.nombre,
                    wait_until_answered=True,
                    ringing_timeout=Duration(seconds=settings.ringing_timeout_s),
                    max_call_duration=Duration(seconds=settings.max_call_duration_s),
                )
            )
            state.contestada = utcnow()
            # El cliente espera la llamada: el agente habla primero y no aguarda ningún "hola".
            # El AMD sigue escuchando al otro lado mientras el saludo suena, para cortar si es buzón.
            saludo = asyncio.create_task(saludar())
            try:
                verdict = await asyncio.wait_for(detector.execute(), AMD_VEREDICTO_S)
            except asyncio.TimeoutError:
                # Sin veredicto a tiempo (lo normal si la persona se queda callada oyendo): es persona.
                verdict = None
    except api.SipCallError as exc:
        state.status_override = status_from_sip_code(exc.sip_status_code)
        state.motivo_fin = f"sip_{exc.sip_status_code or 'error'}"
        logger.info("no se completó el marcado", extra={"call_id": call_id, "sip": exc.sip_status_code})
        ctx.shutdown(reason=state.motivo_fin)
        return
    except Exception:
        logger.exception("error al marcar", extra={"call_id": call_id})
        state.status_override = "fallida"
        state.motivo_fin = "error_al_marcar"
        ctx.shutdown(reason="error_al_marcar")
        return

    if verdict is not None and verdict.is_machine:
        # Es un buzón y el saludo ya está sonando: se corta en seco y se cuelga sin dejar mensaje.
        saludo.cancel()
        try:
            session.interrupt()
        except Exception:
            logger.warning("no se pudo cortar el saludo en el buzón", exc_info=True)
        state.status_override = "buzon"
        state.motivo_fin = str(verdict.category.value)
        await hangup()
        return

    # El saludo ya arrancó; si falló al sintetizarse no se pierde la llamada por eso.
    for resultado in await asyncio.gather(saludo, return_exceptions=True):
        if isinstance(resultado, BaseException):
            logger.exception("falló el saludo inicial", exc_info=resultado)


async def finalize(
    ctx: JobContext,
    settings: Settings,
    call_id: str,
    sale: Sale,
    state: CallState,
    session: AgentSession | None,
    reason: str,
) -> None:
    """Corre al cerrar el job: debe ser rápido (copiar y escribir); la entrega la hace el dispatcher."""
    directory = call_dir(settings.data_dir, call_id)
    if state.status_override:
        estado: CallStatus = state.status_override
    elif state.contestada is None:
        estado = "fallida"
    else:
        estado = evaluate(
            cedula_coincide=state.cedula_coincide,
            celular_confirmado=state.celular_confirmado,
            dijo_acepto=state.dijo_acepto,
        )

    try:
        recorded = ctx.session_directory / "audio.ogg"
        if state.contestada and recorded.exists():
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(recorded, directory / "audio.ogg")
    except Exception:
        logger.exception("no se pudo copiar la grabación", extra={"call_id": call_id})

    write_json_atomic(
        directory / "result.json",
        {
            "estado": estado,
            "cedula_dicha": state.cedula_dicha,
            "cedula_coincide": state.cedula_coincide,
            # Ya no hay pregunta aparte: escuchó las condiciones y dijo "acepto la compra".
            "confirma_condiciones": state.dijo_acepto,
            "correo_confirmado": sale.correo_confirmado,  # lo confirma el cliente en el formulario
            "celular_confirmado": state.celular_confirmado,
            "dijo_acepto": state.dijo_acepto,
            "frase_aceptacion": state.frase_aceptacion,
            "observacion": state.observacion,
            "simulacion": state.simulacion,
            "motivo_fin": state.motivo_fin or reason,
            "contestada": state.contestada.isoformat() if state.contestada else None,
            "finalizada": utcnow().isoformat(),
            "metricas": state.metricas,
            "transcripcion": transcript(session),
        },
    )
    logger.info(
        "llamada finalizada",
        extra={"call_id": call_id, "estado": estado, "venta_id": sale.venta_id, "simulacion": state.simulacion},
    )

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                f"http://{settings.dispatcher_host}:{settings.dispatcher_port}/internal/calls/{call_id}/done",
                headers={"X-Internal-Token": settings.internal_token},
            )
    except Exception:
        # No es grave: el barrido del dispatcher la recoge en menos de un minuto.
        logger.warning("no se pudo avisar al dispatcher", extra={"call_id": call_id})


if __name__ == "__main__":
    cli.run_app(server)
