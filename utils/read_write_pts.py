#读PTS
def read_pts_file(file_path):
    points = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("SIZE"):num_pts = line.split()[1]
            if line.startswith("WIDTH"):width = line.split()[1]
            if line.startswith("HEIGHT"):height = line.split()[1]
            if line.strip() != "" and not line.startswith("VERSION") and not line.startswith("SIZE") and not line.startswith("TYPE") and not line.startswith("DIMENSION") and not line.startswith("POINT_PROPERTIES") and not line.startswith("DATA") and not line.startswith("WIDTH") and not line.startswith("HEIGHT"):
                x, y, _ = line.split()
                points.append((float(x), float(y)))
    return points,int(num_pts),int(width),int(height)

#写PTS
def write_pts_file(file_path, points,width,height):
    with open(file_path, 'w') as f:
        f.write("VERSION: 1\n")
        f.write("SIZE: {}\n".format(len(points)))
        f.write("TYPE: PTS\n")
        f.write("DIMENSION: 2\n")
        f.write("POINT_PROPERTIES: id\n")
        f.write("DATA\n")
        f.write("WIDTH: {}\n".format(width))
        f.write("HEIGHT: {}\n".format(height))
        for i, point in enumerate(points):
            f.write("{:.6f} {:.6f} {}\n".format(point[0], point[1], i+1))
