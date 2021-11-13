import torch
import csv


def main():
    load_path = "logs/absorbing_ffhq_80m_elbo/saved_stats/stats_400000"
    stats = torch.load(load_path)
    val_elbos = stats["val_elbos"]
    step_inc = 5000

    with open("ffhq_80_val_elbo.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "elbo"])
        for idx, val_elbo in enumerate(val_elbos):
            writer.writerow([step_inc * (idx + 1), val_elbo * 20])


if __name__ == "__main__":
    main()
