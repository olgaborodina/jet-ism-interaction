"""
turbulence driving simulation
Code create ICs, param and Config files
This set up is in physical unit.
For dimensionless set up, reset UnitLength=UnitMass=UnitVelocity=1 and BoxSize=1,IsoSoundspeed=1,RhoAve=1

This setup is made for jet launching. We use GAMMA=5./3, turn off reset_temp() routine in AB_TURB, and turn on cooling. 

"""
import sys    
import numpy as np    
import h5py    

#simulation_directory = str(sys.argv[1])
simulation_directory = '.'

""" simulation box parameters """
FloatType = np.float64
IntType = np.int32

# cgs unit
PROTONMASS = FloatType(1.67262178e-24)
BOLTZMANN = FloatType(1.38065e-16)
GRAVITY = FloatType(6.6738e-8) 
PC = FloatType(3.085678e+18)
MYR = FloatType(3.15576e13) 
MSOLAR = FloatType(1.989e+33) 

UnitLength = PC # pc
UnitMass = MSOLAR # Msun
UnitVelocity = FloatType(1e5) # km/s
UnitTime = UnitLength/UnitVelocity
UnitDensity = UnitMass / UnitLength / UnitLength / UnitLength
print ("UnitTime_in_Myr = %.2f"%(UnitTime/MYR))

# simulation set up
GAMMA = 5./3.
BoxSize = FloatType(2000.0) # in code unit
IsoSoundspeed = FloatType(20.0) # in code unit
nH_ISM = 20.0 # in cm^-3
RhoAve = nH_ISM*0.6165*PROTONMASS/UnitDensity # in code unit

NumSnaps = IntType(600)
Ncells = IntType(256) #can do 64 if 128 is too slow
NumberOfCells = IntType( Ncells * Ncells * Ncells )
TimeMax = FloatType(30)#FloatType(1.0*BoxSize/IsoSoundspeed) # in code unit
print ("TimeMax (code unit) = ",TimeMax)

#BH parameters
BH_Hsml = FloatType(90) # r_jet=30
HalfOpeningAngle = FloatType(0)
vTargetJet = (BH_Hsml / 10.)**3
JetDensity = FloatType(1e-26) # g cm^-3
JetDensity_code_units = JetDensity / UnitDensity

""" set up initial conditions """
dx = BoxSize / FloatType(Ncells)
pos_first, pos_last = 0.5 * dx, BoxSize - 0.5 * dx
Grid1d = np.linspace(pos_first, pos_last, Ncells, dtype=FloatType)
xx, yy, zz = np.meshgrid(Grid1d, Grid1d, Grid1d)
Pos = np.zeros([NumberOfCells, 3], dtype=FloatType)
Pos[:,0] = xx.reshape(NumberOfCells)
Pos[:,1] = yy.reshape(NumberOfCells)
Pos[:,2] = zz.reshape(NumberOfCells)
center = np.array([0.5*BoxSize,0.5*BoxSize,0.5*BoxSize])

GridDisplacement = 0.05*dx
np.random.seed(seed=321)
Pos += np.random.uniform(low=-GridDisplacement, high=GridDisplacement,size=Pos.shape)

mTarget = RhoAve*BoxSize**3/NumberOfCells
uTarget = (IsoSoundspeed**2)/(GAMMA-1)/GAMMA
print ("mTarget=%.2f"%mTarget)
print ("uTarget=%.2f"%uTarget)

Mass = np.full(Pos[:,0].shape, mTarget)
Uthermal = np.full(Pos[:,0].shape, uTarget)
Velocity = np.full(Pos.shape, 0.0)

PassiveScalars = np.zeros([Uthermal.shape[0], 1], dtype=np.float64)
radius = np.linalg.norm(Pos-center,axis=-1)
PassiveScalars[radius < BH_Hsml] = 1.0

"""setup turbulence scale"""
ST_scale_length = FloatType(1000.0)


""" write *.hdf5 file; minimum number of fields required by Arepo """

IC = h5py.File(simulation_directory+'/IC.hdf5', 'w')

## create hdf5 groups
header = IC.create_group("Header")
part0 = IC.create_group("PartType0")
part5 = IC.create_group("PartType5")

## header entries
NumPart = np.array([NumberOfCells, 0, 0, 0, 0, 1], dtype = IntType) # no BH here
header.attrs.create("NumPart_ThisFile", NumPart)
header.attrs.create("NumPart_Total", NumPart)
header.attrs.create("NumPart_Total_HighWord", np.zeros(6, dtype = IntType) )
header.attrs.create("MassTable", np.zeros(6, dtype = IntType) )
header.attrs.create("Time", 0.0)
header.attrs.create("Redshift", 0.0)
header.attrs.create("BoxSize", BoxSize)
header.attrs.create("NumFilesPerSnapshot", 1)
header.attrs.create("Omega0", 0.0)
header.attrs.create("OmegaB", 0.0)
header.attrs.create("OmegaLambda", 0.0)
header.attrs.create("HubbleParam", 1.0)
header.attrs.create("Flag_Sfr", 0)
header.attrs.create("Flag_Cooling", 0)
header.attrs.create("Flag_StellarAge", 0)
header.attrs.create("Flag_Metals", 0)
header.attrs.create("Flag_Feedback", 0)
header.attrs.create("Flag_DoublePrecision", 1)

## copy datasets
part0.create_dataset("ParticleIDs", data = np.arange(1, NumberOfCells+1) )
part0.create_dataset("Coordinates", data = Pos)
part0.create_dataset("Masses", data = Mass)
part0.create_dataset("Velocities", data = Velocity)
part0.create_dataset("InternalEnergy", data = Uthermal)
part0.create_dataset("PassiveScalars", data = PassiveScalars)

part5.create_dataset("ParticleIDs", data = np.array([NumberOfCells+1]) )
part5.create_dataset("Masses", data = np.array([1.0]) )
part5.create_dataset("Coordinates", data = np.array([0.5*BoxSize, 0.5*BoxSize, 0.5*BoxSize]).reshape([1,3]) )
part5.create_dataset("Velocities", data = np.array([0.0, 0.0, 0.0]).reshape([1,3]) )


IC.close()


""" set up Config.sh """

ConfigOptions = [ 'VORONOI',
                  'REGULARIZE_MESH_CM_DRIFT',
                  'REGULARIZE_MESH_FACE_ANGLE',
                  'REFINEMENT_SPLIT_CELLS',
                  'REFINEMENT_MERGE_CELLS',
                  'REFINEMENT_VOLUME_LIMIT',
                  'TREE_BASED_TIMESTEPS',
                  'EXTERNALGRAVITY',
                  'DOUBLEPRECISION=1',
                  'OUTPUT_IN_DOUBLEPRECISION',
                  'INPUT_IN_DOUBLEPRECISION',
                  'DOUBLEPRECISION_FFTW',
                  'VORONOI_DYNAMIC_UPDATE',
                  'NO_ISEND_IRECV_IN_DOMAIN',
                  'FIX_PATHSCALE_MPI_STATUS_IGNORE_BUG',
                  'HAVE_HDF5',
                  'AB_TURB',
                  'AB_TURB_NOISOTH',
                  'OUTPUT_DENSITY_GRADIENT',
                  'GAMMA=%g'%GAMMA,
                  'POWERSPEC_GRID=%d'%(Ncells*2),
                  'CHUNKING',
                  'ENLARGE_DYNAMIC_RANGE_IN_TIME',
                  'COOLING',
                  'SHOCK_FINDER_BEFORE_OUTPUT',
                  'BLACK_HOLES',
                  'BH_THERMALFEEDBACK',
                  'BH_DO_NOT_PREVENT_MERGERS',
                  'BH_CONSTANT_POWER_RADIOMODE_IN_ERG_PER_S=1e40',
                  'BH_CONSTANT_POWER_RADIOMODE_TIMELIMIT_IN_MYR=10',
                  'BH_CONSTANT_POWER_RADIOMODE_START_TIME_IN_MYR=15',
                  'BH_JET',
                  'BH_JET_DEBUG',
                  'BH_JET_PASSIVE_SCALAR',
                  'BH_JET_REFINEMENT',
                  'BH_JET_FIXED_DENSITY',
                  'BH_JET_FIXED_INJECTION_RADIUS',
                  'BH_JET_SPHERICAL_JET_REGIONS',
                  'BH_JET_CENTRAL_JET',
                  'BH_JET_VIRTUAL_BUFFER',
                  'BH_FIX_POSITION_AT_BOXHALF',
                  'PASSIVE_SCALARS=1',
                  'OUTPUT_CSND',
                  'OUTPUT_DIVVEL',
                  'OUTPUT_CURLVEL',
                  'OUTPUT_VELOCITY_GRADIENT',
                  'OUTPUT_MACHNUM',
                ]

file = open(simulation_directory + '/Config.sh', 'w')  # create this locally, is then copied over by shell script
for ConfigOption in ConfigOptions:
    file.write(ConfigOption+'\n')
file.close()


""" set up param.txt """

ParamOptions = ['InitCondFile                                      IC',
                'OutputDir                                         output',
                
                'SnapshotFileBase                                  snap',
                'OutputListOn                                      1',
                'OutputListFilename                                OutputList',
                
                'ICFormat                                          3',
                'SnapFormat                                        3',
                
                'TimeLimitCPU                                      150000',
                'CpuTimeBetRestartFile                             14400',
                'ResubmitOn                                        0',
                'ResubmitCommand                                   my-scriptfile',
                
                'MaxMemSize                                        3800',

                'CellShapingSpeed                                  0.5',
                'CellMaxAngleFactor                                2.25',
                
                'TimeBegin                                         0.0',
                'TimeMax                                           %.2f'%TimeMax,  
                'BoxSize                                           %.2f'%BoxSize,
                
                'PeriodicBoundariesOn                              1',
                'ComovingIntegrationOn                             0',
                'CoolingOn                                         1',
                'StarformationOn                                   0',
                
                'Omega0                                            0',
                'OmegaBaryon                                       0',
                'OmegaLambda                                       0',
                'HubbleParam                                       1.0',
                
                'NumFilesPerSnapshot                               1',
                'NumFilesWrittenInParallel                         1',
                'TimeOfFirstSnapshot                               0.0',
                'TimeBetSnapshot                                   %.2f'%(TimeMax/NumSnaps),
                'TimeBetStatistics                                 %.2f'%(TimeMax/NumSnaps),
                'TimeBetTurbSpectrum                               %.2f'%(TimeMax/NumSnaps),
                
                'TypeOfTimestepCriterion                           0',
                'ErrTolIntAccuracy                                 0.025',
                'CourantFac                                        0.4',
                'MaxSizeTimestep                                   %g'%(TimeMax/NumSnaps/3),
                'MinSizeTimestep                                   1e-10',
                
                'MinimumDensityOnStartUp                           0',
                'LimitUBelowThisDensity                            0',
                'LimitUBelowCertainDensityToThisValue              0',
                
                'InitGasTemp                                       0',
                'MinGasTemp                                        10',
                'MinEgySpec                                        0',
                
                'TypeOfOpeningCriterion                            1',
                'ErrTolTheta                                       0.7',
                'ErrTolForceAcc                                    0.0025',
                'MultipleDomains                                   1',
                'TopNodeFactor                                     2',
                
                'DesNumNgb                                         32',
                'MaxNumNgbDeviation                                2',
                
                'UnitVelocity_in_cm_per_s                          %g'%UnitVelocity,
                'UnitLength_in_cm                                  %g'%UnitLength,
                'UnitMass_in_g                                     %g'%UnitMass,
                'GravityConstantInternal                           0',
                
                'ActivePartFracForNewDomainDecomp                  0.3',
                'GasSoftFactor                                     1.5',
                
                'SofteningComovingType0                            0.1',
                'SofteningComovingType1                            0.1',
                'SofteningComovingType2                            0.1',
                'SofteningComovingType3                            0.1',
                'SofteningComovingType4                            0.1',
                'SofteningComovingType5                            0.1',
                'SofteningMaxPhysType0                             0.1',
                'SofteningMaxPhysType1                             0.1',
                'SofteningMaxPhysType2                             0.1',
                'SofteningMaxPhysType3                             0.1',
                'SofteningMaxPhysType4                             0.1',
                'SofteningMaxPhysType5                             0.1',
                'SofteningTypeOfPartType0                          0',
                'SofteningTypeOfPartType1                          0',
                'SofteningTypeOfPartType2                          0',
                'SofteningTypeOfPartType3                          0',
                'SofteningTypeOfPartType4                          0',
                'SofteningTypeOfPartType5                          0',
                
                'IsoSoundSpeed                                     %g'%(IsoSoundspeed),
                'ST_decay                                          %g'%(0.05*ST_scale_length/IsoSoundspeed),
                'ST_energy                                         %g'%(1.0*IsoSoundspeed**3/ST_scale_length/8),
                'ST_DtFreq                                         %g'%(0.005*ST_scale_length/IsoSoundspeed),
                'ST_Kmin                                           %g'%(2*np.pi/ST_scale_length),
                'ST_Kmax                                           %g'%(4*np.pi/ST_scale_length),
                'ST_SolWeight                                      1.',
                'ST_AmplFac                                        1.',
                'ST_Seed                                           42',
                'ST_SpectForm                                      2',
                
                'ReferenceGasPartMass                              %g' % mTarget,
                'TargetGasMassFactor                               1',
                'RefinementCriterion                               1',
                'DerefinementCriterion                             1',
                'MaxVolumeDiff                                     10',
                'MinVolume                                         0',
                'MaxVolume                                         1.0e6',
                'TreecoolFile                                      /n/holystore01/LABS/hernquist_lab/Users/borodina/arepo_rainer/data/TREECOOL_fg_dec11', # path of UVB table for cooling
                'BlackHoleAccretionFactor                          1',
                'BlackHoleFeedbackFactor                           1',
                'BlackHoleEddingtonFactor                          1',
                'SeedBlackHoleMass                                 1',
                'DesNumNgbBlackHole                                128',
                'BlackHoleMaxAccretionRadius                       5000',
                'BlackHoleRadiativeEfficiency                      1',
                'BH_JET_TargetVolume                               %g' % vTargetJet,
                'BlackHoleJetHalfOpeningAngle                      %d' % HalfOpeningAngle,
                'BlackHoleJetDensity                               %g' % JetDensity_code_units,
                'BH_hsml                                           %g' % BH_Hsml
                ]

file = open(simulation_directory + '/param.txt', 'w')  # create this locally, is then copied over by shell script
for ParamOption in ParamOptions:
    file.write(ParamOption+'\n')
file.close()
