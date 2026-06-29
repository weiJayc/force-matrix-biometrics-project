from force_matrix_biometrics.capture import capture_packets
from force_matrix_biometrics.profiles import COUNT_CAPTURE_PROFILE


def main() -> None:
    capture_packets(
        profile=COUNT_CAPTURE_PROFILE,
        output_path="press_img.txt",
        packet_limit=17,
        delay_seconds=3,
        include_index=True,
    )


if __name__ == "__main__":
    main()