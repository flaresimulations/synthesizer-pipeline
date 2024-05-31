#!/bin/bash -l
#SBATCH -J synthesizer_EAGLE_pipeline
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=2
#SBATCH -o logs/job.%J.dump
#SBATCH -e logs/job.%J.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 00:30:00

module purge
# module load rockport-settings
module load gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3 python/3.10.1 
source /cosma7/data/dp004/dc-payy1/my_files/synthesizer/venv_synth/bin/activate

export OMPI_MCA_btl=^openib

tags=(
    000_z020p000
    001_z015p132
    002_z009p993
    003_z008p988
    004_z008p075
    005_z007p050
    006_z005p971
    007_z005p487
    008_z005p037
    009_z004p485
    010_z003p984
    011_z003p528
    012_z003p017
    013_z002p478
    014_z002p237
    015_z002p012
    016_z001p737
    017_z001p487
    018_z001p259
    019_z001p004
    020_z000p865
    021_z000p736
    022_z000p615
    023_z000p503
    024_z000p366
    025_z000p271
    026_z000p183
    027_z000p101
    028_z000p000
)

# Total number of tasks (workers)
Ttasks=$SLURM_NTASKS

volume=L0100N1504
eagle_file=/cosma7/data/Eagle/ScienceRuns/Planck1/${volume}/PE/REFERENCE/data
tag=000_z020p000
outdir=./${volume}/PE/REFERENCE/data/photometry_$tag/
outfile=eagle_subfind_photometry_${tag}

mkdir -p $outdir

echo Tag: $tag
echo $out_file
echo Total number of tasks: $Ttasks
echo CPUs per task: $SLURM_CPUS_PER_TASK
echo Node being used: $SLURM_JOB_PARTITION

grid_name=bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps
grid_dir=/cosma7/data/dp004/dc-payy1/my_files/synthesizer/synthesizer_data/grids/
# numpy memmap location
shm_prefix=/snap7/scratch/dp004/dc-payy1/tmp/
# add if doing multiple redshifts at once 
shm_suffix=$tag
nsegments=5


mpiexec -n $SLURM_NTASKS python3 run_eagle_spectra_shm.py -tag $tag -eagle-file $eagle_file -output $outdir/$outfile -grid-name $grid_name -grid-directory $grid_dir -total-tasks $Ttasks -nthreads $((SLURM_CPUS_PER_TASK)) -shm-prefix $shm_prefix -shm-suffix $shm_suffix -node-name $SLURM_JOB_PARTITION -n-segments $nsegments


echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
