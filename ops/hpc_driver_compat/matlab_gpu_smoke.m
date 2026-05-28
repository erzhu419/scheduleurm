try
    fprintf('MATLAB_VERSION=%s\n', version);
    fprintf('MATLAB_COMPUTER=%s\n', computer);
    fprintf('PCT_LICENSE_TEST=%d\n', license('test', 'Distrib_Computing_Toolbox'));

    g = gpuDevice(1);
    fprintf('GPU_NAME=%s\n', g.Name);
    fprintf('GPU_COMPUTE_CAPABILITY=%s\n', g.ComputeCapability);
    fprintf('GPU_TOTAL_MEMORY=%d\n', g.TotalMemory);
    fprintf('GPU_DRIVER_VERSION=%s\n', g.DriverVersion);
    fprintf('GPU_TOOLKIT_VERSION=%s\n', g.ToolkitVersion);

    x = gpuArray(single([1 2; 3 4]));
    y = gather(x * x);
    fprintf('GPUARRAY_RESULT=%.1f %.1f %.1f %.1f\n', y(1,1), y(1,2), y(2,1), y(2,2));
    reset(g);
    exit(0);
catch ME
    disp(getReport(ME, 'extended', 'hyperlinks', 'off'));
    exit(10);
end
