from fastapi.testclient import TestClient
from main import app  # Importer l'application FastAPI

client = TestClient(app)

def test_predict():
    # Données de test valides pour la prédiction
    test_data = {
        "num__EXT_SOURCE_2": 0.5,
        "num__EXT_SOURCE_3": 0.7,
        "num__DAYS_BIRTH": -12000,
        "num__DAYS_ID_PUBLISH": -3000,
        "num__DAYS_REGISTRATION": -4000,
        "num__SK_ID_CURR": 100001,
        "num__AMT_ANNUITY": 25000.0,
        "num__DAYS_LAST_PHONE_CHANGE": -100,
        "num__DAYS_EMPLOYED": -3650,
        "num__AMT_CREDIT": 500000.0,
        "num__AMT_INCOME_TOTAL": 200000.0,
        "num__REGION_POPULATION_RELATIVE": 0.02,
        "num__AMT_GOODS_PRICE": 450000.0,
        "num__HOUR_APPR_PROCESS_START": 10,
        "num__TOTALAREA_MODE": 0.8,
        "num__AMT_REQ_CREDIT_BUREAU_YEAR": 2.0,
        "num__YEARS_BEGINEXPLUATATION_MEDI": 40.0,
        "num__YEARS_BEGINEXPLUATATION_MODE": 40.0,
        "num__YEARS_BEGINEXPLUATATION_AVG": 40.0,
        "num__OBS_60_CNT_SOCIAL_CIRCLE": 1.0
    }

    # Envoyer la requête POST avec les données de test
    response = client.post("/predict/", json=test_data)

    # Vérifier que la requête est bien acceptée
    assert response.status_code == 200

    # Vérifier que la réponse contient bien les clés attendues
    json_response = response.json()
    assert "credit_decision" in json_response
    assert "repayment_probability" in json_response

    # Vérifier que la probabilité de remboursement est bien comprise entre 0 et 1
    assert 0.0 <= json_response["repayment_probability"] <= 1.0
