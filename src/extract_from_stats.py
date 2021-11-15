import torch
import csv


def main():
    elbo_load_path = "logs/absorbing_ffhq_80m_elbo/saved_stats/stats_400000"
    elbo_stats = torch.load(elbo_load_path)
    elbo_val_elbos = elbo_stats["val_elbos"]

    new_load_path = "logs/absorbing_ffhq_80m_new_with_val_elbo/saved_stats/stats_400000"
    new_stats = torch.load(new_load_path)
    new_val_elbos = new_stats["val_elbos"]

    step_inc = 5000

    with open("ffhq_80_val_elbo.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "new", "elbo"])

        for i in range(len(new_val_elbos)):
            row = [step_inc * (i+1), new_val_elbos[i] * 20, elbo_val_elbos[i] * 20]
            writer.writerow(row)
        # for idx, val_elbo in enumerate(val_elbos):
        #     writer.writerow([step_inc * (idx + 1), val_elbo * 20])


if __name__ == "__main__":
    main()
