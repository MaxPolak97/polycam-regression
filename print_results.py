import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

csv_folder = './csv'
# saliency = 'gradcam'
saliency_list = [ #'gradcam',
                 #'ig',
                 'polycam']
saliency_dict = {
    # 'gradcam-diff-shufflenetv2':'Grad-CAM',
    # 'gradcam-raw-shufflenetv2':'Grad-CAM',
    # 'ig-diff-shufflenetv2': 'Integrated Gradients',
    # 'ig-raw-shufflenetv2': 'Integrated Gradients',
    'pcampm-diff-shufflenetv2-4':'Poly-CAM',
    # 'pcampm-raw-shufflenetv2-4':'Poly-CAM',
                 }

print("faithfulness metrics")
for saliency, saliency_name in saliency_dict.items():
    print(saliency_name)
    print('____________________________________')
    del_auc = pd.read_csv(csv_folder + "/" + "del_auc_" + saliency + ".csv").to_numpy()
    # print(del_auc)
    del_details = pd.read_csv(csv_folder + "/" + "del_details_" + saliency + ".csv").to_numpy()
    del_details_float = np.array(del_details[:, 1:], dtype=float)

    mean_del_auc = np.mean(del_auc[:, 1])
    std_del_auc = np.std(del_auc[:, 1])

    try:
        sens = pd.read_csv(csv_folder + "/" + "sens_" + "_" + saliency + ".csv").to_numpy()
        mean_sens = np.mean(sens[:, 1])
    except:
        mean_sens = 0

    print('Mean Deletion  auc:           %0.3f' % mean_del_auc, '+/-%0.3f' % std_del_auc)
    print('____________________________________')

x = (np.array(range(224))+1)/224

print(x)

fig, ax = plt.subplots(figsize=(10, 8))
fontsize = 22
# ax.set_title("Deletion", fontsize=fontsize)

color = ['red', 'blue', 'green']

for idx_saliency, saliency in enumerate(saliency_dict.keys()):
    del_details = pd.read_csv(csv_folder + "/" + "del_details_" + saliency + ".csv").to_numpy()
    del_auc = pd.read_csv(csv_folder + "/" + "del_auc_" + saliency + ".csv").to_numpy()
    mean_del_details = del_details[:, 1:].mean(0)
    mean_del_auc = np.mean(del_auc[:, 1])

    # Plot mean_del_details
    if idx_saliency == 2:
        ax.plot(x, mean_del_details, label=saliency_dict[saliency] + f' (AUC={round(mean_del_auc, 2)})'
                , color=color[idx_saliency], lw=2)
    else:
        ax.plot(x, mean_del_details, label=saliency_dict[saliency] + f' (AUC={round(mean_del_auc, 2)})'
                , color=color[idx_saliency], lw=2, alpha=1)

    # # Fill the area under the curve
    # ax.fill_between(x, mean_del_details.astype(float), color=color[idx_saliency], alpha=0.2)

# # Add text "AUC" at (0.1, 0.1)
# ax.text(0.2, 0.3, f"AUC={round(mean_del_auc, 2)}", transform=ax.transAxes, fontsize=20)
ax.legend(prop={'size': 21})
plt.ylim((0,1))
plt.suptitle('Faithfullness curve', fontsize=fontsize * 0.9, fontweight="bold") # str(del_details.shape[0]) + ' images '
plt.xlabel('pixels deleted from the input image [%]', fontsize=23)
plt.ylabel(r"$Average\ deletion\ score\ \overline{D}\ [-]$", fontsize=23)
# plt.ylabel("Deletion score D [-]", fontsize=20)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tight_layout()
plt.show()
plt.close()

