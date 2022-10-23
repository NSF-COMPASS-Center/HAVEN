### Deployment script to execute HEV prediction pipeline in ARC

```commandline
sbatch --output <output-folder>/slurm-%j.out deployment/arc/arc_zoonosis_deployment.sh <absolute path to project root directory> <absolute path to configuration file>
```
Slurm executor in ARC replaces '%j' with the Job ID. 

Example:
```bash
sbatch --output output/arc/slurm-%j.out deployment/arc/arc_zoonosis_deployment.sh ~/dev/git/zoonosis/ ~/dev/git/zoonosis/config-files/seed-132197556/hepHostNoTransfer.yaml
```
