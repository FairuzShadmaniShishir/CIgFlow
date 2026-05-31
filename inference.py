import torch
from nanobody_generator import NanobodyGenerator


def load_generator(save_dir="/home/f087s426/Research/Antibody Research/Antigen_Specific Antibody Design/Antibody Sequence Design/Saved Model/temperature_0.7"):
    """Rebuild and load NanobodyGenerator from saved weights."""

    # Load config
    config = torch.load(f"{save_dir}/config.pt")

    # Rebuild architecture with same dims
    generator = NanobodyGenerator(
        seq_dim=config["seq_dim"],
        max_seq_len=config["max_seq_len"],
        hidden_dim=config["hidden_dim"],
    )

    generator.initialize_models(
        merged_embed_dim=config["merged_embed_dim"],
        antigen_embed_dim=config["antigen_embed_dim"],
    )

    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator.flow_model.load_state_dict(
        torch.load(f"{save_dir}/flow_model.pt", map_location=device)
    )
    generator.decoder.load_state_dict(
        torch.load(f"{save_dir}/decoder.pt", map_location=device)
    )
    # generator.encoder.load_state_dict(...)  # if you have one

    # Set to eval mode — IMPORTANT
    generator.flow_model.eval()
    generator.decoder.eval()

    print(f"✅ Model loaded from {save_dir}/")
    return generator


def run_inference(heavy_seq: str, antigen_seq: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load model ──────────────────────────────────────────
    generator = load_generator("/home/f087s426/Research/Antibody Research/Antigen_Specific Antibody Design/Antibody Sequence Design/Saved Model/temperature_0.7")

    # ── 2. Embed your input sequences ─────────────────────────
    # Mirror exactly how train.py builds paired_sequences
    paired = ' '.join(heavy_seq)  # add light seq if needed

    _, _, merged_emb, antigen_emb, _, _ = generator.prepare_dataset(
        [paired], [antigen_seq], batch_size=1
    )

    # ── 3. Generate ────────────────────────────────────────────
    with torch.no_grad():
        new_sequences = generator.generate_multiple_sequences(
            reference_embedding=merged_emb[0],
            antigen_embeddings=antigen_emb[0],
            num_sequences=1000,
            temperature=0.7,
            num_steps=100,
            noise_scale=1.0,
            guidance_scale=0,
            top_k=50,
            top_p=0.9,
            max_len=256,
            batch_size=5,
            show_progress=True,
        )

    return new_sequences

# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import numpy as np
#
#
# def run_trajectory_analysis(heavy_seq: str, antigen_seq: str):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 1. Load model
#     generator = load_generator()
#
#     # 2. Prepare embeddings
#     paired = ' '.join(heavy_seq)
#
#     _, _, merged_emb, antigen_emb, _, _ = generator.prepare_dataset(
#         [paired], [antigen_seq], batch_size=1
#     )
#
#     merged_emb = merged_emb.to(device)
#     antigen_emb = antigen_emb.to(device)
#
#     # 3. Call trajectory sampling 🔥
#     with torch.no_grad():
#         zT, trajectory = generator.flow_model.sample_with_trajectory(
#             merged_emb=merged_emb,
#             antigen_emb=antigen_emb,
#             num_steps=100,
#             guidance_scale=3.0
#         )
#
#     # 4. Decode final latent → sequence
#     decoded = generator.decoder(zT)
#
#     print("Generated latent shape:", zT.shape)
#
#     return decoded, trajectory
# def plot_trajectory(trajectory):
#     traj = torch.stack(trajectory).squeeze().numpy()
#
#     pca = PCA(n_components=2)
#     traj_2d = pca.fit_transform(traj)
#
#     plt.figure()
#     plt.plot(traj_2d[:, 0], traj_2d[:, 1], marker='o')
#     plt.title("Latent Flow Trajectory (PCA)")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.grid()
#     plt.show()

# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":

    # Alternative reference: Sotrovimab VHH (nanobody) - another FDA-approved therapeutic
    # This is a validated single-domain antibody (VHH) targeting SARS-CoV-2
    #antibody = "EVQLVESGGGLVQPGGSLRLSCAASGFTISGYGMSWVRQAPGKGLEWVSSISSSSGYIYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDYGGSYVDYWGQGTLVTVSS"
    # SARS-CoV-2 Spike Protein S-2P (Pfizer/Moderna vaccine version with K986P, V987P stabilization)
    # This is the prefusion-stabilized spike protein used in mRNA vaccines
    #antigen = "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"


    # Trastuzumab (Herceptin) heavy chain variable domain (VH)
    # FDA-approved HER2-targeted therapeutic antibody (Genentech/Roche)
    # Canonical clone sequence used in crystal structures (PDB: 1N8Z)
    antibody = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"

    # HER2 extracellular domain (ECD), domains I-IV
    # Residues 23-645 of UniProt P04626 (mature protein)
    # This is the ectodomain targeted by trastuzumab and pertuzumab
    antigen = "MELAALCRWGLLLALLPPGAASTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLAECRHFDELLVTQNPCTYKITGMAIAIPCINCTGQPILDREAFRIRHPKTPSVQLVHYQMRPGPIPAGPGDREAFRIRHPKTPSVQLVHYQMRPGPIPAGPGDREAFRIRHPKTPSVQLVHYQMRPGPIPAGPGDRDDNPHISGGSTIYNPNYPNLISSVLYNLVTDLDLWMDPETKDEIQQKIGFGKDSQISVTPEGTSAATYLKSCSWLDSGDVNRQFMQRLIKQLTNAGKLDMISQRLNQKNLQYLREQLARRKHSDLIPEGHEQKLISEEDL"

    sequences = run_inference(antibody, antigen)

    #sequences = run_inference(antibody, antigen)

    for i, seq in enumerate(sequences):
        print(f"Sequence {i + 1} (len={len(seq)}): {seq}")

    fasta_file = "Generated Sequence/1000_generated_Antibodies_HER2.fasta"

    with open(fasta_file, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">Sequence_{i + 1}\n")  # FASTA header
            # Wrap sequence every 80 characters (FASTA convention)
            for j in range(0, len(seq), 80):
                f.write(seq[j:j + 80] + "\n")

    print(f"Saved {len(sequences)} sequences to {fasta_file}")


# if __name__ == "__main__":
#
#     antibody = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
#
#     antigen = "MELAALCRWGLLLALLPPGAASTQVCTGTDMKLRLPASPETHLDMLRHLYQGCQVVQGNLELTYLPTNASLSFLQDIQEVQGYVLIAHNQVRQVPLQRLRIVRGTQLFEDNYALAVLDNGDPLNNTTPVTGASPGGLRELQLRSLTEILKGGVLIQRNPQLCYQDTILWKDIFHKNNQLALTLIDTNRSRACHPCSPMCKGSRCWGESSEDCQSLTRTVCAGGCARCKGPLPTDCCHEQCAAGCTGPKHSDCLAECRHFDELLVTQNPCTYKITGMAIAIPCINCTGQPILDREAFRIRHPKTPSVQLVHYQMRPGPIPAGPGDREAFRIRHPKTPSVQLVHYQMRPGPIPAGPGDREAFRIRHPKTPSVQLVHYQMRPGPIPAGPGDRDDNPHISGGSTIYNPNYPNLISSVLYNLVTDLDLWMDPETKDEIQQKIGFGKDSQISVTPEGTSAATYLKSCSWLDSGDVNRQFMQRLIKQLTNAGKLDMISQRLNQKNLQYLREQLARRKHSDLIPEGHEQKLISEEDL"
#
#     # 🔥 NEW: run trajectory analysis
#     decoded, trajectory = run_trajectory_analysis(antibody, antigen)
#
#     # 🔥 Plot trajectory
#     plot_trajectory(trajectory)
#
#     # Optional: still run your normal generation
#     sequences = run_inference(antibody, antigen)
#
#     for i, seq in enumerate(sequences[:5]):
#         print(f"Sequence {i + 1}: {seq}")