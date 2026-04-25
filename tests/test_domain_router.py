from apps.api.app.services.domain_router import DomainRouter
from apps.api.app.services.query_understanding import normalize_ro_text


def route(question: str):
    return DomainRouter().route(normalize_ro_text(question))


def test_demo_salary_question_routes_to_labor_domain():
    result = route("Poate angajatorul să-mi scadă salariul fără act adițional?")

    assert result.legal_domain == "muncă"
    assert result.domain_confidence >= 0.70


def test_fiscal_question_routes_to_fiscal():
    result = route("Cum declar TVA la ANAF pentru o declarație fiscală?")

    assert result.legal_domain == "fiscal"


def test_gdpr_question_routes_to_data_protection():
    result = route("Ce drepturi am pentru date personale conform GDPR?")

    assert result.legal_domain == "protecția datelor"


def test_contravention_question_routes_to_contravention():
    result = route("Cum contest o amendă din proces-verbal contravențional?")

    assert result.legal_domain == "contravențional"


def test_contravention_question_without_diacritics_routes_to_contravention():
    result = route("Cum contest o amenda din proces-verbal contraventional?")

    assert result.legal_domain == "contravențional"


def test_labor_question_without_diacritics_routes_to_labor():
    result = route("Poate angajatorul sa scada salariul fara act aditional?")

    assert result.legal_domain == "muncă"


def test_named_labor_code_routes_to_labor():
    result = route("Ce spune art. 41 din Codul muncii?")

    assert result.legal_domain == "muncă"


def test_civil_question_routes_to_civil():
    result = route("Cum se împarte o moștenire pentru o proprietate?")

    assert result.legal_domain == "civil"


def test_ambiguous_short_question_has_no_domain_or_ambiguity_flag():
    result = route("Ce fac?")

    assert result.legal_domain is None
    assert "low_domain_confidence" in result.ambiguity_flags
