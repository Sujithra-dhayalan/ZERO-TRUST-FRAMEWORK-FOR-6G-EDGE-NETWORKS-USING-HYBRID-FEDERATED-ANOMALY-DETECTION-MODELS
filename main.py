from src.preprocessing import load_and_preprocess_data, create_edge_nodes
from src.models import build_autoencoder, get_rf_classifier
from src.security import ZeroTrustEnforcer
from src.federated_utils import federated_aggregate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(dataset_path):
    # 1. Setup
    data, labels = load_and_preprocess_data(dataset_path)
    nodes = create_edge_nodes(data, labels)
    enforcer = ZeroTrustEnforcer()
    global_ae = build_autoencoder(data.shape[1])
    
    all_node_weights = []
    metrics = {"names": [], "acc": []}

    # 2. Local Training Loop
    for i, (node_df, node_labels) in enumerate(nodes):
        print(f"--- Training Edge Node {i+1} ---")
        X_train, X_test, y_train, y_test = train_test_split(node_df, node_labels, test_size=0.3)

        # Train Autoencoder (Unsupervised)
        local_ae = build_autoencoder(data.shape[1])
        local_ae.fit(X_train, X_train, epochs=5, verbose=0)
        all_node_weights.append(local_ae.get_weights())

        # Train RF (Supervised for XAI/Classification)
        rf = get_rf_classifier()
        rf.fit(X_train, y_train)
        
        # 3. Security Enforcement
        sample = X_test.iloc[:1].to_numpy()
        reconstruction = local_ae.predict(sample, verbose=0)
        mse = np.mean(np.power(sample - reconstruction, 2))
        t_score = enforcer.calculate_trust_score(mse)
        
        acc = accuracy_score(y_test, rf.predict(X_test))
        metrics["names"].append(f"Node {i+1}")
        metrics["acc"].append(acc)

        print(f"Action: {enforcer.access_control(t_score)} | Trust: {t_score:.2f} | Acc: {acc:.2f}")

    # 4. Federated Aggregation
    global_ae.set_weights(federated_aggregate(all_node_weights))
    print("\nGlobal Model Synced.")

    # 5. Visualization
    plt.bar(metrics["names"], metrics["acc"], color='skyblue')
    plt.title("Phase 2: Federated Node Performance")
    plt.show()

if __name__ == "__main__":
    run_simulation('data/Dataset.csv')