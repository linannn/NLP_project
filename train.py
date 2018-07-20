import numpy as np
from xgboost import XGBRegressor

def trainByXgboost(x_train, y_train, x_test, outputFile):
    model = XGBRegressor(
        learn_rate=0.08,
        max_depth=7,
        min_child_weight=4,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1,
        objective='reg:logistic',
        n_estimators=120
    )
    trained_model = model.fit(x_train, y_train)
    y_predict = trained_model.predict(x_test)
    with open(outputFile, 'w') as f:
        for i in y_predict:
            f.write(str(i) + '\n')
