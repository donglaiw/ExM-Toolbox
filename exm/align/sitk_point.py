import SimpleITK as sitk

class sitkPoint:
    # 1. warp points with the transformation
    def __init__(self):
        self.parameter_map = None
        self.transform_map = None
        self.transform_type = None
    
    def setResolution(self, resolution):
        # xyz-order
        self.resolution = resolution

    def setTransformType(self, transform_type, num_iteration = -1):
        self.transform_type = transform_type
        self.parameter_map = self.createParameterMap(transform_type, num_iteration)
        self.elastix.SetParameterMap(self.parameter_map)
    
    def setTransformMap(self, transform_map):
        self.transform_map = transform_map

    #### Setup
    def setPointTransformAffine(self, resolution = None, transform_map = None):
        # https://github.com/SuperElastix/SimpleElastix/issues/91
        # xyz order
        if resolution is None:
            resolution = self.resolution
        if transform_map is None:
            transform_map = self.transform_map
        
        # um -> pix
        param_val = np.array(transform_map["TransformParameters"]).astype(float)
        param_cen = np.array([float(x) for x in transform_map["CenterOfRotationPoint"]])
        self.A_A = param_val[:9].reshape([3,3])        
        self.A_t = (param_val[9:] / resolution).reshape([-1,1])
        self.A_cen = (param_cen / resolution).reshape([-1,1])        
        
    def warpPoint(self, pts, mode = 'backward'):
        # pts: 3xN matrix (xyz-order)
        assert (np.array(pts.shape)==3).sum() > 0
        if pts.shape[0] != 3:
            pts = pts.T

        # A = computeTransformMap(vol_fixed, vol_moving)
        # for pts from vol_moving, need the "inverse" mode to warp it to vol_fixed

        if mode == 'backward':
            pts_out = np.linalg.solve(self.A_A, pts - self.A_cen - self.A_t) + self.A_cen
        elif mode == 'forward':
            pts_out = np.matmul(self.A_A, pts - self.A_cen) + self.A_t + self.A_cen
        return pts_out
        
