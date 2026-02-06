import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    
    # Label Handling
    label_col = 'attack_cat' if 'attack_cat' in df.columns else 'Label'
    if label_col not in df.columns:
        raise ValueError("Target label column not found.")
    
    labels = df[label_col]
    df = df.drop([label_col, 'id'], axis=1, errors='ignore')

    # Handle NaNs
    if labels.isnull().any():
        mask = labels.notna()
        df, labels = df[mask], labels[mask]

    # Encoding & Scaling
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled, labels

def create_edge_nodes(df, labels, num_nodes=3):
    nodes_data = []
    for _ in range(num_nodes):
        node_df, _, node_labels, _ = train_test_split(df, labels, train_size=0.2, stratify=labels)
        nodes_data.append((node_df, node_labels))
    return nodes_data