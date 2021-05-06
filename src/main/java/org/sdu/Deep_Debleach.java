package org.sdu;


import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URISyntaxException;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ij.IJ;

import ij.ImagePlus;
import ij.plugin.filter.PlugInFilter;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;


public class Deep_Debleach implements PlugInFilter {

	private ImagePlus curImage = null;
	
	public static INDArray processImage(INDArray input) {
		Double max = (Double) input.maxNumber();
		Double min = (Double) input.minNumber();
		if (min != 0) {
			input = input.add(-min);
		}
		input = input.div(max);
		int[] shape = new int[] {1,512,512,1};
		input = input.reshape(shape);
		return input;
	}

	public static File toTempFile(InputStream is) throws FileNotFoundException, IOException {
        File f = File.createTempFile("tempfile", ".tmp");
        f.deleteOnExit();


        try (OutputStream os = new BufferedOutputStream(new FileOutputStream(f))) {
            IOUtils.copy(is, os);
            os.flush();
            return f;
        }		
	}
	
	@Override
	public int setup(String arg, ImagePlus imp) {
		// TODO Auto-generated method stub
		curImage = imp;
		return DOES_16+DOES_32+DOES_8G+NO_CHANGES;
	}

	@Override
	public void run(ImageProcessor ip) {
		// TODO Auto-generated method stub
		InputStream modelJsonS;
		InputStream modelWeightsS;
		
		modelJsonS = getClass().getClassLoader().getResourceAsStream("model_config.json");
		modelWeightsS = getClass().getClassLoader().getResourceAsStream("model_weights.h5");
		String modelJson = null;
		String modelWeights = null;
		try {
			modelJson = toTempFile(modelJsonS).getAbsolutePath();
			modelWeights = toTempFile(modelWeightsS).getAbsolutePath();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}



    	ComputationGraph network = null;
		try {
			network = KerasModelImport.importKerasModelAndWeights(modelJson, modelWeights);
		} catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
		/*
		ImagePlus image = IJ.openImage("C:\\Users\\Jacob\\Desktop\\original-bleach9.tif");
		*/
    	INDArray matrix = Nd4j.createFromArray(ip.getFloatArray());
    	
    	INDArray normalized = processImage(matrix);
    	INDArray output = network.outputSingle(normalized);
		int[] shape = new int[] {512,512};
		output = output.reshape(shape);
		
    	FloatProcessor flprocessor = new FloatProcessor(output.toFloatMatrix());
    	ImagePlus result = new ImagePlus(curImage.getTitle() + " - debleached",flprocessor);
    	result.show();				
	}
	
	public static void main(String[] vars) throws URISyntaxException {
		
		
		InputStream modelJsonS;
		InputStream modelWeightsS;
	
		modelJsonS = Deep_Debleach.class.getClassLoader().getResourceAsStream("model_config.json");
		modelWeightsS = Deep_Debleach.class.getClassLoader().getResourceAsStream("model_weights.h5");

		
		
		String modelJson = null;
		String modelWeights = null;
		try {
			modelJson = toTempFile(modelJsonS).getAbsolutePath();
			modelWeights = toTempFile(modelWeightsS).getAbsolutePath();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

    	ComputationGraph network = null;
		try {
			network = KerasModelImport.importKerasModelAndWeights(modelJson, modelWeights);
		} catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
	
		ImagePlus image = IJ.openImage("C:\\Users\\Jacob\\Desktop\\original-bleach9.tif");
		ImageProcessor ip = image.getProcessor();
    	INDArray matrix = Nd4j.createFromArray(ip.getFloatArray());
    	
    	INDArray normalized = processImage(matrix);
    	INDArray output = network.outputSingle(normalized);
		int[] shape = new int[] {512,512};
		output = output.reshape(shape);
    	FloatProcessor flprocessor = new FloatProcessor(output.toFloatMatrix());
    	ImagePlus result = new ImagePlus(image.getTitle() + " - debleached",flprocessor);
    	result.show();				
	}
	
}
