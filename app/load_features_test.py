from .load_data_3d import load_targets, load_samples_inputs
from .load_deviations import load_deviations
from .cut_brain_test import cut_brain
from .feature_test import feature_mean, feature_max, feature_ratio_mean

def load_features(norms=None):
    if norms == None:
        training = True
    else:
        training = False 

    areas = ["whole","lt","mt","rt","lb","mb","rb"]
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
        norms = {}
        data = load_targets()
        print("plotting features")
        for f in features:
            for a in areas:
                feature_inputs , norms["{}_{}".format(f["name"], a)] = f["f"](a)
                data["{}_{}".format(f["name"], a)] = feature_inputs
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
        data = {} 
        for f in features:
            for a in areas:
                feature_inputs = f["f"](a, norms["{}_{}".format(f["name"], a)])
                data["{}_{}".format(f["name"], a)] = feature_inputs
        return data
