import math

import cv2
import exifread
import numpy
from PIL import Image
from pyproj import Proj, transform, Transformer

class Points_Projection(object):
    def __init__(self, cloud_points : list, pic_path : str, camera_martix : list, dis_coeffs : list, show : bool = True,
                 show_scale_percent : int = 30, output : bool = False, output_path : str = ''):
        self.cloud_points = numpy.array(cloud_points)
        self.pic_path = pic_path
        self.camera_matrix = numpy.array(camera_martix)
        self.dis_coeffs = numpy.array(dis_coeffs)
        self.bool_output = output
        self.bool_show = show
        self.output_path = output_path
        self.show_scale_percent = show_scale_percent

    def getXY(self, pic_path: str):
        # pic_path = 'data/pic_01.JPG'
        f = open(pic_path, 'rb')
        tags = exifread.process_file(f)
        if "GPS GPSLatitude" in tags:
            lat_ref = tags["GPS GPSLatitudeRef"]
            lat = tags["GPS GPSLatitude"].printable[1: -1].replace(" ", "").replace("/", ",").split(",")
            lat = float(lat[0]) + float(lat[1]) / 60 + (float(lat[2]) / float(lat[3])) / 3600
            if lat_ref in ["S"]:
                lat = -lat
        if "GPS GPSLongitude" in tags:
            lon_ref = tags["GPS GPSLongitudeRef"]
            lon = tags["GPS GPSLongitude"].printable[1: -1].replace(" ", "").replace("/", ",").split(",")
            lon = float(lon[0]) + float(lon[1]) / 60 + (float(lon[2]) / float(lon[3])) / 3600
            if lon_ref in ["W"]:
                lon = -lon

        if "GPS GPSAltitude" in tags:
            alt = tags["GPS GPSAltitude"].printable.replace(" ", "").replace("/", ",").split(",")
            alt = float(alt[0]) / float(alt[1])

        transrformer = Transformer.from_crs('epsg:4326', 'epsg:4546')
        x, y = transrformer.transform(lat, lon)

        return [y, x, -alt]

    def get_XMP_INf(self, pic_path: str, inf_key: str):
        with Image.open(pic_path) as im:
            xmp_inf = str(im.info.get('xmp')).split('\\n')
            for i in range(0, len(xmp_inf)):
                if len(xmp_inf[i]) == 100:
                    continue
                tmp = xmp_inf[i].split(':')
                tmp[0] = tmp[0].lstrip()
                if tmp[0] == 'drone-dji':
                    key, value = tmp[1].split('=')
                    if key == inf_key:
                        value = value.replace('"', '')
                        try:
                            value = float(value)
                        except Exception as e:
                            print(f"Error converting value to float: {e}")
                        return value

    def process(self):
        roll_angle = math.radians(self.get_XMP_INf(self.pic_path, 'GimbalRollDegree'))
        pitch_angle = math.radians(self.get_XMP_INf(self.pic_path, 'GimbalPitchDegree')+90)
        yaw_angle = math.radians(self.get_XMP_INf(self.pic_path, 'GimbalYawDegree')*(-1))
        print(roll_angle, pitch_angle, yaw_angle)
        Roll = numpy.array([
            [1,                         0,                      0],
            [0,                         math.cos(roll_angle),   -math.sin(roll_angle)],
            [0,                         math.sin(roll_angle),   math.cos(roll_angle)]], dtype=numpy.float32)
        Pitch = numpy.array([
            [math.cos(pitch_angle),     0,                      math.sin(pitch_angle)],
            [0,                         1,                      0],
            [-math.sin(pitch_angle),    0,                      math.cos(pitch_angle)]], dtype=numpy.float32)
        Yaw = numpy.array([
            [math.cos(yaw_angle),       -math.sin(yaw_angle),   0],
            [math.sin(yaw_angle),       math.cos(yaw_angle),    0],
            [0,                         0,                      1]], dtype=numpy.float32)
        rvec = cv2.Rodrigues(Yaw.dot(Pitch).dot(Roll))[0]
        tvec = numpy.array([0, 0, 0], dtype=numpy.float32)
        self.cloud_points -= numpy.array(self.getXY(self.pic_path))

        img_points, jacobian = cv2.projectPoints(self.cloud_points, rvec, tvec, self.camera_matrix, self.dis_coeffs)
        image = cv2.imread(self.pic_path)
        for i in range(len(img_points)):
            cv2.circle(image, (int(img_points[i][0][0]), int(img_points[i][0][1])), 50, (0, 255, 255), -1)

        if self.bool_show:
            width = int(image.shape[1] * self.show_scale_percent / 100)
            height = int(image.shape[0] * self.show_scale_percent / 100)
            dim = (width, height)
            resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow('Resized Image', resized_image)
            cv2.waitKey(0)

        if self.bool_output:
            cv2.imwrite(self.output_path, image)