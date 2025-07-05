import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def create_improved_model(input_shape):
    """Create an improved neural network model with regularization and batch normalization."""
    model = Sequential([
        Dense(32, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='linear')
    ])
    return model

def add_engineered_features(X):
    """Add interaction terms and polynomial features."""
    exp_level = X[:, -1].reshape(-1, 1)  # experience_level is the last column
    cost_of_living = X[:, 0].reshape(-1, 1)  # cost_of_living is the first column
    
    # Add interaction between experience level and cost of living
    interaction = exp_level * cost_of_living
    
    # Add polynomial features for numerical columns
    poly_features = np.column_stack([
        X[:, :3],  # Original numerical features
        X[:, :3] ** 2,  # Squared terms
        interaction  # Interaction term
    ])
    
    return np.column_stack([poly_features, X[:, -1]])  # Add back categorical feature

def main():
    # Load data
    df = pd.read_csv('final_data.csv')
    
    # Define features
    numeric_features = ['cost_of_living', 'infrastructure_level', 'happiness_level']
    categorical_features = ['experience_level']
    experience_order = ['Entry', 'Junior', 'Intermediate', 'Senior', 'Expert']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OrdinalEncoder(categories=[experience_order]), categorical_features)
        ])
    
    # Prepare data
    X = df[numeric_features + categorical_features]
    y = df['mean_salary']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Add engineered features
    X_train_final = add_engineered_features(X_train_processed)
    X_test_final = add_engineered_features(X_test_processed)
    
    # Create and compile model
    model = create_improved_model(X_train_final.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Add callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train_final,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )
    
    # Evaluate model
    train_loss = model.evaluate(X_train_final, y_train, verbose=0)
    test_loss = model.evaluate(X_test_final, y_test, verbose=0)
    
    print(f"\nTraining Loss: {train_loss[0]:.2f}")
    print(f"Test Loss: {test_loss[0]:.2f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 