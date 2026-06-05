import numpy as np

from tensorflow.keras.models import load_model



# ======================================
# LOAD CNN-BILSTM MODEL
# ======================================

model = load_model(

    "../dataset/models/cnn_bilstm_model.h5"
)



# ======================================
# PREDICTION FUNCTION
# ======================================

def predict_bp(ppg_signal):

    try:

        # ==========================
        # CONVERT TO NUMPY
        # ==========================

        ppg_signal = np.array(

            ppg_signal,

            dtype=np.float32
        )


        # ==========================
        # RESHAPE INPUT
        # ==========================

        ppg_signal = ppg_signal.reshape(

            1,
            875,
            1
        )


        # ==========================
        # MODEL PREDICTION
        # ==========================

        prediction = model.predict(

            ppg_signal,

            verbose=0
        )


        # ==========================
        # RETURN BP VALUE
        # ==========================

        predicted_bp = float(

            prediction[0][0]
        )


        return (predicted_bp * 100) + 80


    except Exception as e:

        print("MODEL ERROR:")
        print(e)

        return 120.0
