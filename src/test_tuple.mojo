fn get_values() -> Tuple[Int, Float32]:
    return (42, 3.14)

fn main():
    var result = get_values()
    print("Values:", result[0], result[1])
