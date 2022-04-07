### Test file for json serialization - correct output for ADM json parsing specifications ###

Columns = ["Dessin", "Quality", "Colour"]
Predictions = ["EAN", "EAN", ["EAN", "Price"]]
Certainty = [100, 100, [62, 38]]

json_map = {"Columns": []}
json_pred_cert = {}
print(json_map)
for i in range(len(Columns)):
    json_map["Columns"].append({"Original Class {i}".format(i=i+1): Columns[i],
                                "Model Prediction(s)": []})
    if isinstance(Predictions[i], list):
        json_map["Columns"][i]["Model Prediction(s)"] \
            .append({"Prediction": (Predictions[i]), "Certainty": (Certainty[i])})
    else:
        json_map["Columns"][i]["Model Prediction(s)"]\
            .append({"Prediction": [(Predictions[i])], "Certainty": [(Certainty[i])]})

print(json_map)
