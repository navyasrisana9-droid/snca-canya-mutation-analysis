import urllib.request
from pathlib import Path

def fetch(acc):
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    text = urllib.request.urlopen(url).read().decode()
    return "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))

def write(path, header, seq):
    Path(path).write_text(f">{header}\n{seq}\n")
    print("Wrote", path, "length", len(seq))

Path("data").mkdir(exist_ok=True)

# Amyloid-beta 42 = residues 672-713 of APP (P05067)
app = fetch("P05067")
abeta = app[671:713]
print("Abeta42:", abeta)
assert abeta == "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA", "Abeta numbering check failed"
write("data/abeta42.fasta", "Abeta42 (APP P05067 672-713)", abeta)

# Tau repeat domain = residues 244-372 in the 441-residue isoform (P10636-8)
tau = fetch("P10636-8")
print("Tau length:", len(tau))
assert len(tau) == 441, "Expected the 441-residue tau isoform"
rd = tau[243:372]
print("Tau repeat domain starts:", rd[:14], "ends:", rd[-6:])
assert rd.startswith("QTAPVPMPDLKNVK"), "Tau numbering check failed"
write("data/tau_RD.fasta", "Tau repeat domain 244-372 (P10636-8)", rd)