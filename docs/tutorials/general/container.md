# Containers

Containers package an application and its dependencies into a portable image.
They help make software environments reproducible across laptops, servers, and
computing clusters.

For complete JLab instructions, including available images and site-specific
usage, see the
[JLab Container Guide](https://pages.jlab.org/scicomp/software/jlab-container-docs/).

## Choose a runtime

| Runtime | Best suited for | Notes |
| --- | --- | --- |
| Docker | Local development and CI | Widely supported; normally uses a background daemon. |
| Podman | Local development without a daemon | Mostly compatible with Docker commands and OCI images; supports rootless operation. |
| Apptainer/Singularity | HPC and shared computing systems | Designed to run without elevated privileges and integrate with batch schedulers. |

Apptainer is the community successor to Singularity. Many systems and existing
scripts still use the `singularity` command.

## Docker and Podman

Docker and Podman build images from a `Dockerfile`:

```bash
docker build -t my-image .
docker run --rm -it my-image
```

Podman usually accepts the same command structure:

```bash
podman build -t my-image .
podman run --rm -it my-image
```

Mount a host directory when files must persist after the container exits:

```bash
docker run --rm -it -v /host/path:/work my-image
```

Replace `docker` with `podman` when using Podman.

## Apptainer and Singularity

Apptainer runs single-file images, usually with the `.sif` extension. It can
also pull images from OCI registries:

```bash
apptainer pull my-image.sif docker://ubuntu:24.04
apptainer exec my-image.sif bash
```

Bind host directories that are not mounted automatically:

```bash
apptainer exec --bind /host/path:/work my-image.sif bash
```

On systems using Singularity, substitute `singularity` for `apptainer`.

## Good practices

- Use versioned image tags instead of `latest` for reproducible work.
- Keep important data outside the container and mount it at runtime.
- Treat images as immutable; rebuild them when dependencies change.
- Avoid embedding credentials, tokens, or private keys in images.
- Confirm the runtime and storage policies of a shared cluster before building
  or running images.
