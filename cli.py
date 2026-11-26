from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True)

OutputDir = Annotated[str, typer.Argument(help="Directory to store the result.")]
WriteFk = Annotated[
    bool,
    typer.Option(
        "--write-foreign-key-index",
        "-fk",
        help="Write foreign key indices on all tables.",
    ),
]
KeepZip = Annotated[
    bool,
    typer.Option(
        "--keep-zip-file",
        "-k",
        help="Keep the zip file downloaded from PDOK after extraction.",
    ),
]
TimeOut = Annotated[
    int,
    typer.Option(
        "--timeout",
        "-t",
        help="Timeout for downloading files.",
    ),
]
ChunkSize = Annotated[
    int,
    typer.Option(
        "--chunk-size",
        "-c",
        help="Size in bytes of the chunks used to stream the download to disk.",
    ),
]


@app.command()
def download_bhrgt_geopackage(
    output_dir: OutputDir,
    write_foreign_key_index: WriteFk = False,
    keep_zip_file: KeepZip = False,
    timeout: TimeOut = 60,
    chunk_size_mb: ChunkSize = 1 << 20,
):
    """
    Download the BRO geotechnical borehole (BHR-GT) GeoPackage from PDOK.

    Creates an index on the BRO id and optionally on all foreign keys.

    """
    from geost.download import pdok_geopackage

    typer.echo("Downloading BHR-GT geopackage")
    gpkg = pdok_geopackage.download(
        "BHR-GT",
        output_dir,
        delete_zip=not keep_zip_file,
        chunk_size_mb=chunk_size_mb,
        timeout=timeout,
    )

    typer.echo("Create index idx_bro_id")
    pdok_geopackage.create_bro_index(
        gpkg,
        'CREATE INDEX IF NOT EXISTS "idx_bro_id" ON "borehole_research" ("bro_id")',
    )

    if write_foreign_key_index:
        pdok_geopackage.write_fk_index(gpkg)

    typer.echo(f"Saved BHR-GT geopackage in {output_dir}")


@app.command()
def download_bhrg_geopackage(
    output_dir: OutputDir,
    write_foreign_key_index: WriteFk = False,
    timeout: TimeOut = 60,
    chunk_size_mb: ChunkSize = 1 << 20,
):
    """
    Download the BRO geological borehole (BHR-G) GeoPackage from PDOK.

    Creates an index on the BRO id and optionally on all foreign keys.

    """
    from geost.download import pdok_geopackage

    typer.echo("Downloading BHR-G geopackage")
    gpkg = pdok_geopackage.download(
        "BHR-G", output_dir, timeout=timeout, chunk_size_mb=chunk_size_mb
    )

    typer.echo("Create index idx_bro_id")
    pdok_geopackage.create_bro_index(
        gpkg,
        'CREATE INDEX IF NOT EXISTS "idx_bro_id" ON "borehole_research" ("bro_id")',
    )

    if write_foreign_key_index:
        pdok_geopackage.write_fk_index(gpkg)

    typer.echo(f"Saved BHR-G geopackage in {output_dir}")


@app.command()
def download_bhrp_geopackage(
    output_dir: OutputDir,
    write_foreign_key_index: WriteFk = False,
    keep_zip_file: KeepZip = False,
    timeout: TimeOut = 60,
    chunk_size_mb: ChunkSize = 1 << 20,
):
    """
    Download the BRO pedological borehole (BHR-P) GeoPackage from PDOK.

    Creates an index on the BRO id and optionally on all foreign keys.

    """
    from geost.download import pdok_geopackage

    typer.echo("Downloading BHR-P geopackage")
    gpkg = pdok_geopackage.download(
        "BHR-P",
        output_dir,
        delete_zip=not keep_zip_file,
        timeout=timeout,
        chunk_size_mb=chunk_size_mb,
    )

    typer.echo("Create index idx_bro_id")
    pdok_geopackage.create_bro_index(
        gpkg,
        'CREATE INDEX IF NOT EXISTS "idx_bro_id" ON "borehole_research" ("bro_id")',
    )

    if write_foreign_key_index:
        pdok_geopackage.write_fk_index(gpkg)

    typer.echo(f"Saved BHR-P geopackage in {output_dir}")


@app.command()
def download_cpt_geopackage(
    output_dir: OutputDir,
    write_foreign_key_index: WriteFk = False,
    keep_zip_file: KeepZip = False,
    timeout: TimeOut = 60,
    chunk_size_mb: ChunkSize = 1 << 25,
):
    """
    Download the BRO cone penetration test (CPT) GeoPackage from PDOK.

    Creates an index on the BRO id and optionally on all foreign keys.

    """
    from geost.download import pdok_geopackage

    typer.echo("Downloading CPT geopackage")
    gpkg = pdok_geopackage.download(
        "CPT",
        output_dir,
        delete_zip=not keep_zip_file,
        timeout=timeout,
        chunk_size_mb=chunk_size_mb,
    )

    typer.echo("Create index idx_bro_id")
    pdok_geopackage.create_bro_index(
        gpkg,
        'CREATE INDEX IF NOT EXISTS "idx_bro_id" ON "geotechnical_cpt_survey" ("bro_id")',
    )

    if write_foreign_key_index:
        pdok_geopackage.write_fk_index(gpkg)

    typer.echo(f"Saved CPT geopackage in {output_dir}")


@app.command()
def download_sfr_geopackage(
    output_dir: OutputDir,
    keep_zip_file: KeepZip = False,
    timeout: TimeOut = 60,
    chunk_size_mb: ChunkSize = 1 << 20,
):
    """
    Download the BRO soil face research (SFR) GeoPackage from PDOK.

    """
    from geost.download import pdok_geopackage

    typer.echo("Downloading SFR geopackage")
    pdok_geopackage.download(
        "SFR",
        output_dir,
        delete_zip=not keep_zip_file,
        timeout=timeout,
        chunk_size_mb=chunk_size_mb,
    )
    typer.echo(f"Saved SFR geopackage in {output_dir}")
