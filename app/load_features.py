from .load_data_3d import load_targets, load_samples_inputs
from .cut_brain import cut_brain
from .feature import feature_mean, feature_max, feature_ratio_mean

def load_features(norms=None):
    if norms == None:
        training = True
    else:
        training = False 

    areas = ["lt","mt","rt","lb","mb","rb"]
    inputs = [
        {
            "area": "whole",
            "val": load_samples_inputs(training)
        }
    ]
    for a in areas:
        inputs.append(
            {
                "area": a,
                "val": cut_brain(inputs[0]["val"], a)
            }
            )
    features = [
        {
            "name": "mean",
            "f": feature_mean
        },
        {
            "name": "ratio_mean",
            "f": feature_ratio_mean
        },
        {
            "name": "max",
            "f": feature_max
        },
    ]
    if norms == None:
        norms = []
        data = load_targets()
        print("plotting features")
        for f in features:
            for i in inputs:
                feature_inputs , norms["{}_{}".format(f["name"], i["area"])] = f["f"](i["val"])
                data["{}_{}".format(f["name"], i["area"])] = feature_inputs
               # plt.figure()
               # plt.scatter(
               #     feature_inputs,
               #     data["Y"].tolist(),
               # )
               # plt.savefig("plots/line_{}_{}.pdf".format(
               #     f["name"], i["area"]
               # ))
               # plt.close()
        return data, norms
    else:
        data = [] 
        for f in features:
            for i in inputs:
                feature_inputs = f["f"](i["val"], norms["{}_{}".format(f["name"], i["area"])])
                data["{}_{}".format(f["name"], i["area"])] = feature_inputs
        return data
