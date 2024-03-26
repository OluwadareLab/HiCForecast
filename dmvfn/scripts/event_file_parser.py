from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

#input_file = "./../logs/hic_train_log/20240304-001515/validate/events.out.tfevents.1709529315.a23b7e2a54c9.14033.1"
#out_file = "./../results/dflogs/20240304-001515.csv"
input_file = "./../logs/hic_train_log/20240304-001809/validate/events.out.tfevents.1709529489.f214aa2340b9.27732.1"
out_file = "./../results/dflogs/20240304-001809.csv"
names = ["hic disco_35_0", "hic disco_35_1", "hic disco_35_2", "hic ssim_35_0", "hic ssim_35_1", "hic ssim_35_2", "hic pearson_35_0", 
        "hic pearson_35_1", "hic pearson_35_2", "hic psnr_35_0", "hic psnr_35_1", "hic psnr_35_2", "hic hicrep_34_0", "hic hicrep_34_1", 
        "hic hicrep_34_2"]
print("Loading file")
event_acc = EventAccumulator(input_file)
print("Reloading to beginning")
event_acc.Reload()
x = pd.DataFrame()
for name in names:
    print("Working on ", name)
    temp_dict = event_acc.Scalars(name)
    current_list = []
    for step in range(300):
        v = temp_dict[step].value
        current_list.append(round(v, 5))
    x[name[4:]] = current_list
x.to_csv(out_file, sep=',', index=True, encoding='utf-8')
