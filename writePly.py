def write_ply_file(points, filename):
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".format(len(points))
    with open(filename, 'w') as f:
        f.write(header)
        for point in points:
            f.write("{} {} {} {} {} {}\n".format(point[0,0], point[0,1], point[0,2], point[0,3], point[0,4], point[0,5]))

