from datetime import date, datetime

import pytest

from voice_confirm.domain import (
    BOGOTA,
    InvalidPhone,
    Sale,
    build_instructions,
    conditions_text,
    cuotas_a_letras,
    evaluate,
    fecha_a_letras,
    is_contact_allowed,
    normalize_co_phone,
    numero_a_letras,
    parse_holidays,
    pesos_a_letras,
    safe_room_fragment,
    status_from_sip_code,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3001234567", "+573001234567"),
        ("300 123 4567", "+573001234567"),
        ("+57 300-123-4567", "+573001234567"),
        ("573001234567", "+573001234567"),
        ("6019165459", "+576019165459"),
    ],
)
def test_normalize_co_phone(raw: str, expected: str) -> None:
    assert normalize_co_phone(raw) == expected


@pytest.mark.parametrize("raw", ["12345", "2001234567", "+1 415 555 0100", ""])
def test_normalize_co_phone_rejects(raw: str) -> None:
    with pytest.raises(InvalidPhone):
        normalize_co_phone(raw)


def bogota(y: int, m: int, d: int, h: int, mi: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=BOGOTA)


def test_contact_hours_weekday() -> None:
    # 2026-10-27 es martes
    assert is_contact_allowed(bogota(2026, 10, 27, 7, 0), set())
    assert is_contact_allowed(bogota(2026, 10, 27, 18, 59), set())
    assert not is_contact_allowed(bogota(2026, 10, 27, 6, 59), set())
    assert not is_contact_allowed(bogota(2026, 10, 27, 19, 0), set())


def test_contact_hours_saturday_sunday_holiday() -> None:
    assert is_contact_allowed(bogota(2026, 10, 31, 8, 0), set())  # sábado
    assert not is_contact_allowed(bogota(2026, 10, 31, 15, 0), set())
    assert not is_contact_allowed(bogota(2026, 11, 1, 10, 0), set())  # domingo
    holidays = parse_holidays("2026-11-02, 2026-11-16")
    assert not is_contact_allowed(bogota(2026, 11, 2, 10, 0), holidays)  # lunes festivo
    assert is_contact_allowed(bogota(2026, 11, 3, 10, 0), holidays)


@pytest.mark.parametrize(
    ("n", "text"),
    [
        (0, "cero"),
        (16, "dieciséis"),
        (21, "veintiuno"),
        (35, "treinta y cinco"),
        (100, "cien"),
        (101, "ciento uno"),
        (1000, "mil"),
        (21000, "veintiún mil"),
        (80000, "ochenta mil"),
        (480000, "cuatrocientos ochenta mil"),
        (1000000, "un millón"),
        (1500000, "un millón quinientos mil"),
        (2026, "dos mil veintiséis"),
        (21000000, "veintiún millones"),
    ],
)
def test_numero_a_letras(n: int, text: str) -> None:
    assert numero_a_letras(n) == text


def test_pesos_y_cuotas() -> None:
    assert pesos_a_letras(480000) == "cuatrocientos ochenta mil pesos"
    assert pesos_a_letras(2000000) == "dos millones de pesos"
    assert pesos_a_letras(21) == "veintiún pesos"
    assert cuotas_a_letras(1) == "una cuota"
    assert cuotas_a_letras(6) == "seis cuotas"
    assert cuotas_a_letras(21) == "veintiuna cuotas"


def test_fecha_a_letras() -> None:
    assert fecha_a_letras(date(2026, 10, 5)) == "cinco de octubre de dos mil veintiséis"
    assert fecha_a_letras(date(2026, 11, 1)) == "primero de noviembre de dos mil veintiséis"


def make_sale(**overrides: object) -> Sale:
    data: dict[str, object] = {
        "venta_id": "GX-000123",
        "telefono": "300 123 4567",
        "nombre": "Carlos Pérez",
        "cedula": "1.234.567.890",
        "programa": "Inteligencia Artificial para Empresas",
        "valor_total": 480000,
        "numero_cuotas": 6,
        "valor_cuota": 80000,
        "fecha_primera_cuota": "2026-11-05",
    }
    data.update(overrides)
    return Sale.from_dict(data)


def test_sale_roundtrip_normalizes() -> None:
    sale = make_sale()
    assert sale.telefono == "+573001234567"
    assert sale.cedula == "1234567890"
    assert Sale.from_dict(sale.to_dict()) == sale


def test_conditions_text_reads_amounts_in_words() -> None:
    text = conditions_text(make_sale())
    assert "cuatrocientos ochenta mil pesos" in text
    assert "seis cuotas mensuales de ochenta mil pesos" in text
    assert "cinco de noviembre de dos mil veintiséis" in text
    assert "480" not in text  # nada de cifras: la voz las lee mal


def test_conditions_text_single_installment() -> None:
    text = conditions_text(make_sale(numero_cuotas=1, valor_cuota=480000))
    assert "una cuota mensual de cuatrocientos ochenta mil pesos" in text


def test_instructions_never_include_registered_cedula() -> None:
    assert "1234567890" not in build_instructions(make_sale())


def test_evaluate() -> None:
    def ev(c, t, a):
        return evaluate(cedula_coincide=c, celular_confirmado=t, dijo_acepto=a)

    assert ev(True, True, True) == "confirmada"
    assert ev(False, None, None) == "revision"
    assert ev(True, False, None) == "revision"
    assert ev(True, True, False) == "revision"
    assert ev(True, True, None) == "incompleta"
    assert ev(None, None, None) == "incompleta"


def test_call_validates_phone_but_never_reads_email() -> None:
    from voice_confirm.domain import acceptance_prompt, celular_a_voz, purchase_text

    assert celular_a_voz("+573001234567") == "tres cero cero, uno dos tres, cuatro cinco seis siete"
    sale = make_sale(correo="ana@gmail.com", correo_confirmado=True)
    assert sale.correo_confirmado is True
    text = purchase_text(sale)
    assert text.startswith(conditions_text(sale))
    assert "confirme si su número de celular es tres cero cero, uno dos tres" in text
    assert "correo" not in text and "ana" not in text
    assert "ana@gmail.com" not in build_instructions(sale)
    assert acceptance_prompt(sale) == f"Para terminar, diga en voz alta: yo, {sale.nombre}, acepto la compra."


def test_repregunta_is_only_the_short_question() -> None:
    from voice_confirm.domain import phone_repregunta

    sale = make_sale()
    assert phone_repregunta(sale) == "¿Me confirma si su número de celular es tres cero cero, uno dos tres, cuatro cinco seis siete?"
    assert "pesos" not in phone_repregunta(sale)


def test_instructions_let_tools_speak() -> None:
    text = build_instructions(make_sale())
    assert "Di exactamente" not in text
    assert "NO digas nada" in text


def test_status_from_sip_code() -> None:
    assert status_from_sip_code(486) == "ocupado"
    assert status_from_sip_code(480) == "no_contesta"
    assert status_from_sip_code(408) == "no_contesta"
    assert status_from_sip_code(403) == "fallida"
    assert status_from_sip_code(None) == "fallida"


def test_safe_room_fragment() -> None:
    assert safe_room_fragment("GX-000123") == "GX-000123"
    assert safe_room_fragment("a/b c") == "a-b-c"
    assert safe_room_fragment("///") == "venta"


def test_parece_ingles_detecta_narracion_del_modelo():
    from voice_confirm.domain import parece_ingles

    assert parece_ingles('The customer said "1, 2, 3, 4, 5, 6" as their cédula.')
    assert parece_ingles("I'll register those digits.")
    assert parece_ingles("Voy a llamar a registrar_cedula.")
    assert not parece_ingles("Hola, le habla el sistema automático de confirmación de Genix Academy.")
    assert not parece_ingles("¿Confirma estas condiciones? Diga sí o no.")
    assert not parece_ingles("Usted compró el programa Inteligencia Artificial para Empresas.")
    assert not parece_ingles("Listo.")
