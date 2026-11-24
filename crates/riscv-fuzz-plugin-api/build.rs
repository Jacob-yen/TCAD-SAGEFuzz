fn main() {
    let protoc =
        protoc_bin_vendored::protoc_bin_path().expect("failed to locate vendored protoc binary");

    let mut config = prost_build::Config::new();
    config.protoc_executable(protoc);
    config.include_file("protocol_generated.rs");
    config.protoc_arg("--experimental_allow_proto3_optional");
    config
        .compile_protos(&["proto/plugin.proto"], &["proto"])
        .expect("failed to compile plugin protocol schema");
}
