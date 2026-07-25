from pathlib import Path
import json

# Full 102-class name list
CLASS_NAMES = [
    "rice leaf roller", "rice leaf caterpillar", "paddy stem maggot", "asiatic rice borer",
    "yellow rice borer", "rice gall midge", "Rice Stemfly", "brown plant hopper",
    "white backed plant hopper", "small brown plant hopper", "rice water weevil",
    "rice leafhopper", "grain spreader thrips", "rice shell pest", "grub", "mole cricket",
    "wireworm", "white margined moth", "black cutworm", "large cutworm", "yellow cutworm",
    "red spider", "corn borer", "army worm", "aphids", "Potosiabre vitarsis", "peach borer",
    "english grain aphid", "green bug", "bird cherry-oataphid", "wheat blossom midge",
    "penthaleus major", "longlegged spider mite", "wheat phloeothrips", "wheat sawfly",
    "cerodonta denticornis", "beet fly", "flea beetle", "cabbage army worm", "beet army worm",
    "Beet spot flies", "meadow moth", "beet weevil", "sericaorient alismots chulsky",
    "alfalfa weevil", "flax budworm", "alfalfa plant bug", "tarnished plant bug",
    "Locustoidea", "lytta polita", "legume blister beetle", "blister beetle",
    "therioaphis maculata Buckton", "odontothrips loti", "Thrips", "alfalfa seed chalcid",
    "Pieris canidia", "Apolygus lucorum", "Limacodidae", "Viteus vitifoliae",
    "Colomerus vitis", "Brevipoalpus lewisi McGregor", "oides decempunctata",
    "Polyphagotars onemus latus", "Pseudococcus comstocki Kuwana", "parathrene regalis",
    "Ampelophaga", "Lycorma delicatula", "Xylotrechus", "Cicadella viridis", "Miridae",
    "Trialeurodes vaporariorum", "Erythroneura apicalis", "Papilio xuthus",
    "Panonchus citri McGregor", "Phyllocoptes oleiverus ashmead", "Icerya purchasi Maskell",
    "Unaspis yanonensis", "Ceroplastes rubens", "Chrysomphalus aonidum",
    "Parlatoria zizyphus Lucus", "Nipaecoccus vastalor", "Aleurocanthus spiniferus",
    "Tetradacus c Bactrocera minax", "Dacus dorsalis(Hendel)", "Bactrocera tsuneonis",
    "Prodenia litura", "Adristyrannus", "Phyllocnistis citrella Stainton",
    "Toxoptera citricidus", "Toxoptera aurantii", "Aphis citricola Vander Goot",
    "Scirtothrips dorsalis Hood", "Dasineura sp", "Lawana imitata Melichar",
    "Salurnis marginella Guerr", "Deporaus marginatus Pascoe", "Chlumetia transversa",
    "Mango flat beak leafhopper", "Rhytidodera bowrinii white", "Sternochetus frigidus",
    "Cicadellidae",
]
NAME_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Confusion clusters found in test_confusion_matrix.csv
CONFUSION_CLUSTERS = {
    "cutworm": ["black cutworm", "large cutworm", "yellow cutworm"],
    "plant_bug": ["tarnished plant bug", "alfalfa plant bug", "Miridae"],
    "blister_beetle": ["blister beetle", "legume blister beetle", "lytta polita"],
    "leafhopper": ["Cicadellidae", "Cicadella viridis"],
    "army_worm": ["beet army worm", "cabbage army worm"],
    "plant_hopper": ["brown plant hopper", "white backed plant hopper", "small brown plant hopper"],
    "aphid": ["aphids", "english grain aphid", "bird cherry-oataphid", "green bug",
              "Toxoptera citricidus", "Toxoptera aurantii"],
}

# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IP102_ROOT = PROJECT_ROOT / "datasets" / "raw" / "Classification" / "ip102_v1.1"
ANNOTATION_FILES = {
    "train": IP102_ROOT / "train.txt",
    "val": IP102_ROOT / "val.txt",
    "test": IP102_ROOT / "test.txt",
}


def build_subset(annotation_file: Path, target_names: list[str], output_file: Path):

    target_idx = {NAME_TO_IDX[name] for name in target_names}
    old_to_new = {old: new for new, old in enumerate(sorted(target_idx))}

    counts = {name: 0 for name in target_names}
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(annotation_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            image_name, label = line.split()
            label = int(label)
            if label in target_idx:
                new_label = old_to_new[label]
                fout.write(f"{image_name} {new_label}\n")
                counts[CLASS_NAMES[label]] += 1
                n_written += 1

    label_map = [
        {"new_label": new, "original_label": old, "class_name": CLASS_NAMES[old]}
        for old, new in sorted(old_to_new.items(), key=lambda kv: kv[1])
    ]
    return n_written, counts, label_map


def main():
    for cluster_name, target_names in CONFUSION_CLUSTERS.items():
        out_dir = IP102_ROOT / "subsets" / cluster_name
        print(f"\n=== cluster: {cluster_name} ({', '.join(target_names)}) ===")

        label_map = None
        for split, ann_path in ANNOTATION_FILES.items():
            if not ann_path.exists():
                print(f"  [skip] {ann_path} not found")
                continue
            out_path = out_dir / f"{split}.txt"
            n_written, counts, label_map = build_subset(ann_path, target_names, out_path)
            print(f"  {split}: wrote {n_written} lines -> {out_path}")
            for name, c in counts.items():
                print(f"      {name:30s} n={c}")

        if label_map is not None:
            with open(out_dir / "label_map.json", "w") as f:
                json.dump(label_map, f, indent=2)
            print(f"  label_map.json -> {out_dir / 'label_map.json'}")


if __name__ == "__main__":
    main()