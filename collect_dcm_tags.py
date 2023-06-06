import os, pydicom, argparse

#tags included: 
#1) equipment descriptive, 
#2) procedure/sequence descriptive, 
#3) image descriptive,
#4) anatomy descriptive

#tags excluded: 
#1) id of patient/study/series/sequence/lesion/device etc.;
#2) sequence tags
#3) number/index in sequence tags
#4) manufacturer specific tags
#5) algorithm description tags (i.e image compression,reconstruction,etc.)
#6) ROI/overlay tags
#7) Advanced equipment description tags (mainly DX)

#1.general and MR image tags
#2.CT image tags
#3.US image tags
#4.DX/XR/CR image tags
#5.PET image tags
#6.MG image tags


tags=[\
      'Modality','Manufacturer','StudyDescription','SeriesDescription','ManufacturerModelName',\
        'BodyPartExamined','ScanningSequence','SequenceVariant','MRAcquisitionType',\
        'SequenceName','ScanOptions','SliceThickness','RepetitionTime','EchoTime','InversionTime',\
        'MagneticFieldStrength','NumberOfPhaseEncodingSteps','EchoTrainLength','PercentSampling',\
        'PercentPhaseFieldOfView','PixelBandwidth','AcquisitionMatrix','ImageType',\
        'FlipAngle','VariableFlipAngleFlag','PatientPosition','PhotometricInterpretation','Rows',\
        'Columns','PixelSpacing','ContrastBolusVolume','ContrastBolusTotalDose',\
        'ContrastBolusIngredient','ContrastBolusIngredientConcentration',\
        'PatientOrientation','ImageLaterality','ImageComments','ImagePositionPatient',\
        'ImageOrientationPatient','SamplesPerPixel','PhotometricInterpretation',\
        'PlanarConfiguration','PixelAspectRatio','BitsAllocated','BitsStored','HighBit',\
        'PixelRepresentation','ColorSpace','AngioFlag','ImagingFrequency','EchoNumbers',\
        'SpacingBetweenSlices','TriggerTime','NominalInterval','BeatRejectionFlag','LowRRValue',\
        'HighRRValue','IntervalsAcquired','PVCRejection','SkipBeats','HeartRate','TriggerWindow',\
        'ReconstructionDiameter','ReceiveCoilName','TransmitCoilName','InPlanePhaseEncodingDirection',\
        'SAR','dBdt', 'B1rms', 'TemporalPositionIdentifier', 'NumberOfTemporalPositions', 'TemporalResolution',\
        'SliceProgressionDirection','IsocenterPosition', \
         \
         'KVP','DataCollectionDiameter','DistanceSourceToDetector','DistanceSourceToPatient',\
         'GantryDetectorTilt','TableHeight','RotationDirection','ExposureTime','XRayTubeCurrent','Exposure',\
         'ImageAndFluoroscopyAreaDoseProduct','FilterType','GeneratorPower','FocalSpots','ConvolutionKernel',\
         'WaterEquivalentDiameter','RevolutionTime','SingleCollimationWidth','TotalCollimationWidth',\
         'TableSpeed','TableFeedPerRotation','SpiralPitchFactor','DataCollectionCenterPatient',\
         'ReconstructionTargetCenterPatient','ExposureModulationType','EstimatedDoseSaving',\
         'CTDIvol','CalciumScoringMassFactorPatient','CalciumScoringMassFactorDevice','EnergyWeightingFactor',\
         'MultienergyCTAcquisition','AcquisitionNumber','RescaleIntercept','RescaleSlope',\
         'PatientSupportAngle','TableTopLongitudinalPosition','TableTopLateralPosition',\
         'TableTopPitchAngle','TableTopRollAngle',\
      \
      'StageName','StageNumber','NumberOfStages','ViewName','ViewNumber','NumberOfEventTimers',\
      'NumberOfViewsInStage','EventElapsedTimes','EventTimerNames','HeartRate','IVUSAcquisition','IVUSPullbackRate','IVUSGatedRate',\
      'TransducerType','FocusDepth','MechanicalIndex','BoneThermalIndex','CranialThermalIndex',\
      'SoftTissueThermalIndex','SoftTissueFocusThermalIndex','DepthOfScanField',\
      \
      'ExposureInuAs','AcquisitionDeviceProcessingDescription','AcquisitionDeviceProcessingCode',\
      'CassetteOrientation','CassetteSize','ExposuresOnPlate','RelativeXRayExposure','ExposureIndex',\
      'TargetExposureIndex','DeviationIndex','Sensitivity','PixelSpacingCalibrationType','PixelSpacingCalibrationDescription',\
      'DerivationDescription','AcquisitionDeviceProcessingDescription','AcquisitionDeviceProcessingCode',\
      'RescaleType','WindowCenterWidthExplanation','CalibrationImage','PresentationLUTShape',\
     \
      'PlateID','CassetteID','FieldOfViewShape','FieldOfViewDimensions','ImagerPixelSpacing',\
      'ExposureIndex','TargetExposureIndex','DeviationIndex','Sensitivity','DetectorConditionsNominalFlag',\
      'DetectorTemperature','DetectorType','DetectorConfiguration','DetectorDescription','DetectorMode',\
      'DetectorBinning','DetectorElementPhysicalSize','DetectorElementSpacing','DetectorActiveShape',\
      'DetectorActiveDimensions','DetectorActiveOrigin','DetectorManufacturerName','DetectorManufacturerModelName',\
      'FieldOfViewOrigin','FieldOfViewRotation','FieldOfViewHorizontalFlip','PixelSpacingCalibrationType',\
      'PixelSpacingCalibrationDescription',\
      \
      'PrimaryPromptsCountsAccumulated','SecondaryCountsAccumulated','SliceSensitivityFactor',\
      'DecayFactor','DoseCalibrationFactor','ScatterFractionFactor','DeadTimeFactor','IsocenterPosition',\
      'PatientGantryRelationshipCodeSequence','TriggerSourceOrType','CardiacFramingType','PVCRejection',\
      'CollimatorGridName','CollimatorType','CorrectedImage','TypeOfDetectorMotion','Units','CountsSource',\
      'ReprojectionMethod','SUVType','RandomsCorrectionMethod','RandomsCorrectionMethod','DecayCorrection',\
      'ReconstructionMethod','DetectorLinesOfResponseUsed','ScatterCorrectionMethod','ScatterCorrectionMethod',\
      'AxialMash','TransverseMash','CoincidenceWindowWidth','SecondaryCountsType',\
      \
      'PositionerType','PositionerPrimaryAngle','PositionerSecondaryAngle','PositionerPrimaryAngleDirection',\
      'ImageLaterality','BreastImplantPresent','PartialView','PartialViewDescription','OrganExposed'\
     ]

tags=list(set(tags))
tags.sort()


def get_first_files(dir_in,dir_out,tags):
    ndirs=0
    for root, dirs, files in os.walk(dir_in):
        if files:
            first_file = os.path.join(root, files[0])
            try:
                if pydicom.misc.is_dicom(first_file):
                    if ndirs % 1000 == 0: print('{} DICOM files written'.format(ndirs))
                    ndirs=ndirs+1                                
                    #print('found DICOM file:',first_file)
                    ds=pydicom.filereader.dcmread(first_file,stop_before_pixels=True,specific_tags=tags)
                    outfil=dir_out+'/'+pydicom.uid.generate_uid()+'.dcm'                    
                    #print('writing:',outfil)
                    pydicom.filewriter.dcmwrite(outfil,ds)
            except Exception as e:
                print('error reading',first_file)
                
                
def get_parser():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Traverse input directory recursively. For each subdirectory, pick one file and image-specific tags into target directory.')

    # Positional arguments.
    parser.add_argument("dir_in", help="Input directory")
    parser.add_argument("dir_out", help="Output directory")    
    return parser.parse_args()
    
    
if __name__ == "__main__":
    p = get_parser()
    print('Traversing',p.dir_in)
    get_first_files(p.dir_in,p.dir_out,tags)            
    print('Done.')