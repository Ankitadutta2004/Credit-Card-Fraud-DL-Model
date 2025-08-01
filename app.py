{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e74b8768-02ca-42bf-83ff-ea9a718c2ef4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-29 12:37:05.531 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\ankit\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "from keras.models import load_model\n",
    "\n",
    "# ============================\n",
    "# 🎯 Load Model and Threshold\n",
    "# ============================\n",
    "@st.cache_resource\n",
    "def load_artifacts():\n",
    "    model = load_model('autoencoder_model.h5')\n",
    "    threshold = joblib.load('best_threshold.pkl')\n",
    "    mse_min = joblib.load('mse_min.pkl')\n",
    "    mse_max = joblib.load('mse_max.pkl')\n",
    "    return model, threshold, mse_min, mse_max\n",
    "\n",
    "autoencoder, threshold, mse_min, mse_max = load_artifacts()\n",
    "\n",
    "# ============================\n",
    "# 📊 Streamlit UI\n",
    "# ============================\n",
    "st.set_page_config(page_title=\"Credit Card Fraud Detector\", layout=\"centered\")\n",
    "st.title(\"💳 Credit Card Fraud Detection\")\n",
    "st.write(\"Upload credit card transaction data (CSV) to detect fraud using Autoencoder.\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Upload CSV file\", type=[\"csv\"])\n",
    "\n",
    "if uploaded_file:\n",
    "    df = pd.read_csv(uploaded_file)\n",
    "    if 'Class' in df.columns:\n",
    "        df = df.drop('Class', axis=1)  # Drop labels if present\n",
    "\n",
    "    st.subheader(\"📄 Preview of Uploaded Data\")\n",
    "    st.dataframe(df.head())\n",
    "\n",
    "    # ============================\n",
    "    # 🔍 Run Fraud Detection\n",
    "    # ============================\n",
    "    st.subheader(\"🔎 Detecting Fraud...\")\n",
    "\n",
    "    # Predict\n",
    "    preds = autoencoder.predict(df)\n",
    "    mse = np.mean(np.power(df - preds, 2), axis=1)\n",
    "\n",
    "    # Normalize errors\n",
    "    probs = (mse - mse_min) / (mse_max - mse_min)\n",
    "    y_pred = [1 if e > threshold else 0 for e in mse]\n",
    "\n",
    "    # Results\n",
    "    df_result = df.copy()\n",
    "    df_result['Anomaly Score'] = probs\n",
    "    df_result['Fraud Prediction'] = y_pred\n",
    "\n",
    "    st.success(\"✅ Detection complete!\")\n",
    "\n",
    "    st.subheader(\"📋 Detection Results\")\n",
    "    st.dataframe(df_result.head(10))\n",
    "\n",
    "    fraud_count = sum(y_pred)\n",
    "    legit_count = len(y_pred) - fraud_count\n",
    "    st.metric(label=\"🚨 Fraudulent Transactions\", value=fraud_count)\n",
    "    st.metric(label=\"✅ Legit Transactions\", value=legit_count)\n",
    "\n",
    "    # Download Results\n",
    "    csv = df_result.to_csv(index=False).encode('utf-8')\n",
    "    st.download_button(\"Download Results as CSV\", data=csv, file_name=\"fraud_predictions.csv\", mime='text/csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "30d7355b-1e09-4da0-8ec0-5a13ab33c3ad",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
