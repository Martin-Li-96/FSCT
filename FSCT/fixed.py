import os

def fix_ply_file(input_file, output_file):
    header = []
    header_size = 0
    vertex_size_bytes = 0
    with open(input_file, "rb") as f:
        while True:
            line = f.readline()
            header_size += len(line)
            if line.startswith(b"property"):
                parts = line.strip().split()
                if len(parts) == 3:
                    type_str = parts[1].decode()
                    if type_str == "float":
                        vertex_size_bytes += 4
                    elif type_str == "double":
                        vertex_size_bytes += 8
                    elif type_str in ("uchar", "uint8"):
                        vertex_size_bytes += 1
                    elif type_str in ("int", "int32"):
                        vertex_size_bytes += 4
                    else:
                        raise ValueError(f"Unsupported property type: {type_str}")
            if b"end_header" in line:
                header.append(line)
                break
            elif line.startswith(b"element vertex"):
                # placeholder, will replace later
                header.append(b"ELEMENT_VERTEX_PLACEHOLDER\n")
            else:
                header.append(line)

        # binary data after header
        rest = f.read()

    total_file_size = os.path.getsize(input_file)
    vertex_data_size = len(rest)
    num_vertices = vertex_data_size // vertex_size_bytes
    leftover = vertex_data_size % vertex_size_bytes

    print(f"Header size: {header_size}")
    print(f"Vertex size: {vertex_size_bytes}")
    print(f"Vertices: {num_vertices}")
    print(f"Leftover bytes (will be dropped): {leftover}")

    # rebuild header with correct vertex count
    fixed_header = []
    for line in header:
        if line == b"ELEMENT_VERTEX_PLACEHOLDER\n":
            fixed_header.append(f"element vertex {num_vertices}\n".encode())
        else:
            fixed_header.append(line)

    # write fixed file (without leftover junk)
    with open(output_file, "wb") as out_f:
        out_f.writelines(fixed_header)
        out_f.write(rest[:num_vertices * vertex_size_bytes])

# Example usage
fix_ply_file("ROBSON_2023_raycloud.ply", "fixed_file.ply")

