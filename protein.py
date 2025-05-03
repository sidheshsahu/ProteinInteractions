import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import joblib
import os

app = Flask(__name__)

model = joblib.load('interactions.joblib')

def clean_sequence(sequence):
    standard_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return ''.join([aa for aa in sequence if aa in standard_amino_acids])

def extract_properties(sequence):
    cleaned_sequence = clean_sequence(sequence)
    analysis = ProteinAnalysis(cleaned_sequence)
    properties = {
        "Molecular Weight": analysis.molecular_weight(),
        "Isoelectric Point": analysis.isoelectric_point(),
        "Aromaticity": analysis.aromaticity(),
        "Instability Index": analysis.instability_index(),
        "Hydrophobicity": analysis.gravy(),
        "ph_5.5": analysis.charge_at_pH(5.5),
        "ph_7": analysis.charge_at_pH(7),
        "ph_8.5": analysis.charge_at_pH(8.5),
        "Length": len(cleaned_sequence),
    }
    return properties

def predict_interaction_logic(protein1, protein2, model):
    p1_features = extract_properties(protein1)
    p2_features = extract_properties(protein2)

    data = {
        "Protein_1_Molecular_Weight": p1_features["Molecular Weight"],
        "Protein_1_Isoelectric_Point": p1_features["Isoelectric Point"],
        "Protein_1_Aromaticity": p1_features["Aromaticity"],
        "Protein_1_Instability_Index": p1_features["Instability Index"],
        "Protein_1_Hydrophobicity": p1_features["Hydrophobicity"],
        "Protein_1_chargeAT_ph5.5": p1_features["ph_5.5"],
        "Protein_1_chargeAT_ph7": p1_features["ph_7"],
        "Protein_1_chargeAT_ph8.5": p1_features["ph_8.5"],
        "Protein_1_len": p1_features["Length"],
        "Protein_2_Molecular_Weight": p2_features["Molecular Weight"],
        "Protein_2_Isoelectric_Point": p2_features["Isoelectric Point"],
        "Protein_2_Aromaticity": p2_features["Aromaticity"],
        "Protein_2_Instability_Index": p2_features["Instability Index"],
        "Protein_2_Hydrophobicity": p2_features["Hydrophobicity"],
        "Protein_2_chargeAT_ph5.5": p2_features["ph_5.5"],
        "Protein_2_chargeAT_ph7": p2_features["ph_7"],
        "Protein_2_chargeAT_ph8.5": p2_features["ph_8.5"],
        "Protein_2_len": p2_features["Length"],
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        protein1 = request.form.get('protein1', '')
        protein2 = request.form.get('protein2', '')

        if not protein1 or not protein2:
            return render_template('index.html', error="Both protein sequences are required.")

        try:
            pred, prob = predict_interaction_logic(protein1, protein2, model)
            return render_template('index.html', interaction=bool(pred), confidence=round(prob, 4))
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
