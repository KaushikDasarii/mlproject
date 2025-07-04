import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initate_model_training(self, train_array, test_array):
    try:
        logging.info('Splitting Dependent and Independent variables')
        xtrain, ytrain, xtest, ytest = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )

        # Base models (untuned)
        base_models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "GradientBoosting Regressor": GradientBoostingRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor()
        }

        # Default evaluation
        model_report = evaluate_models(xtrain, ytrain, xtest, ytest, base_models)
        logging.info(f"Base Model Report: {model_report}")
        print("Base Model R² Scores:")
        print(model_report)

        # Hyperparameter tuning configurations
        tuned_models = {}
        
        for name in base_models:
            logging.info(f"Tuning started for {name}")

            if name == "Linear Regression":
                tuned_models[name] = LinearRegression()

            elif name == "Lasso":
                grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
                tuned_models[name] = GridSearchCV(Lasso(), grid, cv=5, scoring='r2', n_jobs=-1).fit(xtrain, ytrain).best_estimator_

            elif name == "Ridge":
                grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
                tuned_models[name] = GridSearchCV(Ridge(), grid, cv=5, scoring='r2', n_jobs=-1).fit(xtrain, ytrain).best_estimator_

            elif name == "K-Neighbors Regressor":
                grid = {'n_neighbors': list(range(2, 31))}
                tuned_models[name] = GridSearchCV(KNeighborsRegressor(), grid, cv=5, scoring='r2', n_jobs=-1).fit(xtrain, ytrain).best_estimator_

            elif name == "Decision Tree":
                grid = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
                tuned_models[name] = GridSearchCV(DecisionTreeRegressor(), grid, cv=5, scoring='r2', n_jobs=-1).fit(xtrain, ytrain).best_estimator_

            elif name == "Random Forest Regressor":
                grid = {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
                tuned_models[name] = GridSearchCV(RandomForestRegressor(), grid, cv=3, scoring='r2', n_jobs=-1).fit(xtrain, ytrain).best_estimator_

            elif name == "XGBRegressor":
                param_dist = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
                tuned_models[name] = RandomizedSearchCV(XGBRegressor(), param_dist, n_iter=5, scoring='r2', cv=3, n_jobs=-1).fit(xtrain, ytrain).best_estimator_

            elif name == "CatBoosting Regressor":
                param_dist = {'depth': [4, 6, 8], 'learning_rate': [0.01, 0.03], 'iterations': [300, 500]}
                tuned_models[name] = RandomizedSearchCV(CatBoostRegressor(verbose=False), param_dist, n_iter=5, scoring='r2', cv=3, n_jobs=-1).fit(xtrain, ytrain).best_estimator_

            elif name == "GradientBoosting Regressor":
                param_dist = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05], 'max_depth': [3, 5]}
                tuned_models[name] = RandomizedSearchCV(GradientBoostingRegressor(), param_dist, n_iter=5, scoring='r2', cv=3, n_jobs=-1).fit(xtrain, ytrain).best_estimator_

            elif name == "AdaBoost Regressor":
                param_dist = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.05, 0.1]}
                tuned_models[name] = RandomizedSearchCV(AdaBoostRegressor(), param_dist, n_iter=5, scoring='r2', cv=3, n_jobs=-1).fit(xtrain, ytrain).best_estimator_

            logging.info(f"Tuning complete for {name}")

        # Evaluate tuned models
        tuned_model_scores = {}
        for name, model in tuned_models.items():
            ypred = model.predict(xtest)
            score = r2_score(ytest, ypred)
            tuned_model_scores[name] = score

        print("\nTuned Model R² Scores:")
        print(tuned_model_scores)
        logging.info(f"Tuned Model R² Scores: {tuned_model_scores}")

        best_model_name = max(tuned_model_scores, key=tuned_model_scores.get)
        best_model = tuned_models[best_model_name]
        best_score = tuned_model_scores[best_model_name]

        print(f"\n Best Tuned Model: {best_model_name} with R² Score: {best_score}")
        logging.info(f"Best Tuned Model: {best_model_name} with R² Score: {best_score}")

        # Save the best model
        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )
        logging.info('Best tuned model saved to disk')

        # Final evaluation
        ytest_pred = best_model.predict(xtest)
        mae, rmse, r2 = model_metrics(ytest, ytest_pred)

        return mae, rmse, r2

    except Exception as e:
        logging.error("Exception occurred during training and tuning")
        raise CustomException(e, sys)
