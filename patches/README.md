# Reconstructing the simulator

The differentiable physics simulator and base evolutionary operators used in
this work derive from ELDiR:

> Strgar L, Matthews D, Hummer T, Kriegman S. Evolution and learning in
> differentiable robots. Proceedings of Robotics: Science and Systems, 2024.
> https://github.com/lstrgar/ELDiR

Those files are **not redistributed here**. `eldir-to-divcoll.patch` contains
only our modifications to them, expressed as a diff against the ELDiR baseline
commit `bcd88b08`.

Run `../setup_simulator.sh` from the repository root, or do it by hand:

```sh
git clone https://github.com/lstrgar/ELDiR.git /tmp/eldir
cd /tmp/eldir
git checkout bcd88b08
git apply /path/to/this/repo/patches/eldir-to-divcoll.patch
cd /path/to/this/repo
mkdir -p simulator
cp /tmp/eldir/simulator/sim.py           simulator/
cp /tmp/eldir/simulator/utils.py         simulator/
cp /tmp/eldir/operators/defaults/geno_pheno.py operators/defaults/
cp /tmp/eldir/operators/defaults/mutate.py     operators/defaults/
cp /tmp/eldir/operators/defaults/select.py     operators/defaults/
cp /tmp/eldir/utils/disk_utils.py        utils/
cp /tmp/eldir/utils/optim_utils.py       utils/
```

The patch has been verified to apply cleanly to `bcd88b08` and to reproduce
the seven files byte-for-byte.

Note that the figure scripts in `figures/` read the CSV files in
`figures/csvs/` directly. **Every figure and table in the paper can be
regenerated without the simulator at all.**
