import numpy as np
import pandas as pd
import onnxruntime as rt
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier

def preprocess_data(data):
    f_take = data.columns.tolist()
    print(len(f_take))

    f_deletion = ['competentie_vakdeskundigheid_toepassen', 'pla_historie_ontwikkeling', 'afspraak_inspanningsperiode',
                         'afspraak_laatstejaar_aantal_woorden', 'afspraak_aantal_woorden', 'afspraak_signaal_voor_medewerker',
                         'persoonlijke_eigenschappen_uitstroom_verw_vlgs_klant', 'persoonlijke_eigenschappen_taaleis_voldaan',
                        'competentie_kwaliteit_leveren', 'competentie_gedrevenheid_en_ambitie_tonen',
                         'persoonlijke_eigenschappen_opstelling', 'competentie_overtuigen_en_beïnvloeden', 'competentie_aansturen',
                         'competentie_other', 'competentie_op_de_behoeften_en_verwachtingen_van_de__klant__richten', 'persoonlijke_eigenschappen_houding_opm',
                         'competentie_materialen_en_middelen_inzetten', 'persoonlijke_eigenschappen_leergierigheid_opm', 'competentie_formuleren_en_rapporteren',
                         'persoonlijke_eigenschappen_initiatief_opm', 'competentie_onderzoeken', 'persoonlijke_eigenschappen_presentatie_opm',
                         'competentie_met_druk_en_tegenslag_omgaan', 'persoonlijke_eigenschappen_communicatie_opm', 'persoonlijke_eigenschappen_doorzettingsvermogen_opm',
                         'competentie_instructies_en_procedures_opvolgen', 'competentie_leren', 'competentie_omgaan_met_verandering_en_aanpassen',
                         'persoonlijke_eigenschappen_flexibiliteit_opm', 'persoonlijke_eigenschappen_zelfstandigheid_opm','Ja','Nee']
    
    f_take = [x for x in f_take if x not in f_deletion]
    print(len(f_take))
    df_selected = data[f_take]
    print(df_selected.shape)
    return df_selected



def test_model(model,X,y,X_train, X_test, y_train, y_test):

    if isinstance(model, Pipeline):
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        cross_val_scores = cross_val_score(model, X, y, cv=5)
    # Case 2: If the model is an ONNX Runtime session
    elif isinstance(model, rt.InferenceSession):
        # Perform inference using ONNX runtime
        onnx_predictions = model.run(None, {'X': X_test.values.astype(np.float32)})
        y_pred = onnx_predictions[0]
        y_proba = onnx_predictions[0]  
        cross_val_scores = None  
    else:
        raise ValueError("Model is neither a scikit-learn pipeline nor an ONNX model session.")

    #pipe.fit(X_train, y_train)
    #y_pred = pipe.predict(X_test)
    #y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba) if y_proba is not None else ([], [], [])
    roc_auc = auc(fpr, tpr) if y_proba is not None else None
    
    if y_proba is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    stats = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'ROC AUC': roc_auc,
            'Cross-Validation Scores': cross_val_scores,
        }
    return stats

'''
data = pd.read_csv('C:\\Users\\91948\\Desktop\\SE\\Software-Testing-Project\\investigation_train_large_checked.csv')
data = preprocess_data(data)
selector = VarianceThreshold()
classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
pipeline = Pipeline(steps=[('feature selection', selector), ('classification', classifier)])
statistics = test_model(pipeline,data)
print(statistics)
'''