Columns = ["Dessin", "Quality", "Colour"]
Predictions = ["EAN", "EAN", ["EAN", "Price"]]
Certainty = [100, 100, [62, 38]]

json_map = {"Columns": []}
json_pred_cert = {}
print(json_map)
for i in range(len(Columns)):
    json_map["Columns"].append({"Original Class {i}".format(i=i+1): Columns[i],
                                "Model Prediction(s)": []})
    json_map["Columns"][i]["Model Prediction(s)"]\
            .append({"Prediction": (Predictions[i]), "Certainty": (Certainty[i])})

print(json_map)
#print(json_map["Columns"][0]["Original Class 1"])

#class_prediction = {"Prediction": [], "Certainty": []}
#original_class = {"Original Class": [], "Model Prediction(s)": []}
# column_map = {}
# class_predictions = {}
# for i in range(len(Columns)):
#     column_map["Original Class {i}".format(i=i+1)] = Columns[i]
#     multiple_predictions_pred = []
#     multiple_predictions_cert = []
#     for n in range(len(Predictions)):
#         # print(isinstance(Predictions[n], list))
#         # if isinstance(Predictions[n], list):
#         #     for m in range(len(Predictions[n])):
#         #         multiple_predictions_pred.append(Predictions[n][m])
#         #         multiple_predictions_cert.append(Certainty[n][m])
#         #     class_predictions["Prediction {i}".format(i=i+1)] = multiple_predictions_pred
#         #     class_predictions["Certainty of Prediction {i}".format(i=i + 1)] = multiple_predictions_cert
#         # else:
#             class_predictions["Majority Prediction"] = Predictions[0]
#             class_predictions["Certainty of Prediction"] = Certainty[0]
#
#     column_map["Model Predictions Class {i}".format(i=i+1)] = [class_predictions]
# print(class_predictions)
#
# json_object = {"Columns": [column_map]}
#
# print(json_object)


#class_prediction["Prediction"] = "EAN"
#class_prediction["Certainty"] = 100
#print(class_prediction)


#original_class["Original Class"] = "Dessin"
#original_class["Model Prediction(s)"] = [class_prediction, class_prediction]

#print(original_class)

#json_object = {"Columns": []}
#json_object["Columns"] = [original_class]

#print(json_object)




#json_object["Columns"] = [{"Original Class": [], "Model Prediction(s)": []}]
#json_object["Original Class"][0] = "Dessin"