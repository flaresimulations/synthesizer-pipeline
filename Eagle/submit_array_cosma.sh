#!/bin/bash -l
#SBATCH -J synthesizer_EAGLE_pipeline
####SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G 
#SBATCH -o logs_array/job.%J.dump
#SBATCH -e logs_array/job.%J.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --array=0
###SBATCH --no-requeue
#SBATCH --exclusive=user
#SBATCH -t 00:10:00

module purge
# module load rockport-settings
module load gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3 python/3.10.1 
source /cosma7/data/dp004/dc-payy1/my_files/synthesizer/venv_synth/bin/activate

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

# array=(973 974 1043 1044 1112 1113 1182 1183 1251 1252
#         1321 1322 1390 1391 1392 1393 1460 1461 1462 1463 
#         1530 1531 1532 1533)

volume=REF_100
eagle_file=/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data
tag=000_z020p000
outdir=./output/${volume}/$tag/
outfile=eagle_subfind_photometry_${tag}

mkdir -p $outdir

echo Tag: $tag
echo $out_file
echo Number of CPUs: $((SLURM_NTASKS*SLURM_CPUS_PER_TASK))
echo CPUs per task: $SLURM_CPUS_PER_TASK

grid_name=bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps
grid_dir=/cosma7/data/dp004/dc-payy1/my_files/synthesizer/synthesizer_data/grids/

python3 run_eagle_spectra.py -tag $tag -eagle-file $eagle_file -output $outdir/$outfile -grid-name $grid_name -grid-directory $grid_dir -chunk $SLURM_ARRAY_TASK_ID -nthreads $((SLURM_CPUS_PER_TASK)) -node-name $SLURM_JOB_PARTITION

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

