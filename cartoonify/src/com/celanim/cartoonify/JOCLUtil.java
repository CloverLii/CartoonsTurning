
package com.celanim.cartoonify;

import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_DEVICE_VERSION;
import static org.jocl.CL.CL_PLATFORM_NAME;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetDeviceInfo;
import static org.jocl.CL.clGetPlatformInfo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

/**
 * An utility class provides several basic functions to get platform and device
 * information.
 */
public final class JOCLUtil {
	
	/**
	 * Read the resource file as a string
	 * 
	 * 
	 * @param filename
	 * @return
	 * @throws IOException
	 */
	public static String readResourceToString(String filename) {
		//
		InputStream inputStream = Cartoonify.class.getResourceAsStream(filename);
		if (inputStream == null) {
			System.err.println("Cannot load src/" + filename);
			System.exit(1);
		}
		StringBuilder resultStringBuilder = new StringBuilder(); // String builder is faster than string 
		try (BufferedReader br = new BufferedReader(new InputStreamReader(inputStream))) {
			String line;// Read one line and append it to the output string builder
			while ((line = br.readLine()) != null) {
				resultStringBuilder.append(line + "\n");
			}
		} catch (IOException e) {
			System.err.println("Error occurs when reading src/"+ filename);
		}
		return resultStringBuilder.toString();
	}

	/**
	 * Obtains all the platforms on the machine.
	 * 
	 * @return an array of platform ids
	 */
	public static cl_platform_id[] getAllPlatforms() {
		// Obtain the number of platform on this machine.
		int[] num_platforms = new int[1];
		CL.clGetPlatformIDs(0, null, num_platforms);
		int numPlatforms = num_platforms[0];
		//System.out.println("Number of CLPlatforms " + numPlatforms);

		// Obtain all platform ID
		cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
		CL.clGetPlatformIDs(numPlatforms, platforms, null);

		return platforms;
	}
	
	/**
	 * Obtains all devices of given type on a specific 'platform'
	 * 
	 * @param platform   platform id
	 * @param deviceType device type
	 * @return an array of device ids
	 */
	public static cl_device_id[] getAllDevices(cl_platform_id platform, long deviceType) {
		// Obtain all devices on this 'platform'
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];
		//System.out.println("Number of devices in '" + getPlatformInfoString(platform, CL_PLATFORM_NAME) + "' : " + numDevices);
		// Obtain all device IDs
		cl_device_id[] devices = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, deviceType, numDevices, devices, null);

		// Print out device information
		for (cl_device_id device : devices) {
			// Get device name (CL_DEVICE_NAME)
			String deviceName = getDeviceInfoString(device, CL_DEVICE_NAME);
			//System.out.printf("%s:\t%s\n", device, deviceName);
			// To show more device details, please refer to
			// http://www.jocl.org/samples/JOCLDeviceQuery.java
		}

		return devices;
	}

	/**
	 * Returns the value of the platform info parameter with the given name
	 *
	 * @param platform  The platform
	 * @param paramName The parameter name
	 * @return The value
	 */
	public static String getPlatformInfoString(cl_platform_id platform, int paramName) {
		// Obtain the length of the string that will be queried
		long size[] = new long[1];
		clGetPlatformInfo(platform, paramName, 0, null, size);

		// Create a buffer of the appropriate size and fill it with the info
		byte buffer[] = new byte[(int) size[0]];
		clGetPlatformInfo(platform, paramName, buffer.length, Pointer.to(buffer), null);

		// Create a string from the buffer (excluding the trailing \0 byte)
		return new String(buffer, 0, buffer.length - 1);
	}
	
	/**
	 * Returns the value of the device info parameter with the given name
	 *
	 * @param device    The device
	 * @param paramName The parameter name
	 * @return The value
	 * 
	 * Reference: http://www.jocl.org/samples/JOCLDeviceQuery.java
	 */
	public static String getDeviceInfoString(cl_device_id device, int paramName) {
		// Obtain the length of the string that will be queried
		long size[] = new long[1];
		clGetDeviceInfo(device, paramName, 0, null, size);

		// Create a buffer of the appropriate size and fill it with the info
		byte buffer[] = new byte[(int) size[0]];
		clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

		// Create a string from the buffer (excluding the trailing \0 byte)
		return new String(buffer, 0, buffer.length - 1);
	}
	
    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    public static String getString(cl_device_id device, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }
    
    /**
     * Returns the OpenCL version of the given device, as a float
     * value
     *
     * @param device The device
     * @return The OpenCL version
     */
    public static float getOpenCLVersion(cl_device_id device)
    {
        String deviceVersion = getString(device, CL_DEVICE_VERSION);
        String versionString = deviceVersion.substring(7, 10);
        float version = Float.parseFloat(versionString);
        return version;
    }
	
}
