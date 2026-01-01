#!/usr/bin/env python3

import argparse
import requests
import pathlib
import logging
import json
import pandas as pd
from typing import Union
from collections import OrderedDict
from request_jlog import fetch_jlog_entries

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(args):
    try:
        config = fetch_nps_entries(args.run)
    except Exception as e:
        logger.fatal(f"Fatal error: {e}")
        exit(1)

    if not args.quiet:
        logger.info("Fetched JLog entry:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

    fadc_config_urls = [
        url for url in sorted(config["attachment_urls"]) if "nps-vme" in url
    ]
    vtp_config_urls = [
        url for url in sorted(config["attachment_urls"]) if "nps-vtp" in url
    ]

    assert (
        len(fadc_config_urls) == 5
    ), f"Expected 5 FADC config URLs, got {len(fadc_config_urls)}"
    assert (
        len(vtp_config_urls) == 5
    ), f"Expected 5 VTP config URLs, got {len(vtp_config_urls)}"

    vme_configs = []  # list of dicts
    vtp_configs = []  # list of dicts
    for i, url in enumerate(fadc_config_urls, start=1):
        logger.info(f"parsing FADC Attachment {i}: {url} ...")
        vme_configs.append(parse_vme_config(url))

    for i, url in enumerate(vtp_config_urls, start=1):
        logger.info(f"parsing VTP Attachment {i}: {url} ...")
        vtp_configs.append(parse_vtp_config(url))

    # combine the configurations from different crates
    channel_map = pd.read_csv(args.channel_map, delimiter=r"\s+")
    channels = channel_map["index"].tolist()
    unique_slots = channel_map["slot"].unique().tolist()

    vme_config_dict = {ch: {} for ch in channels}
    vtp_config_dict = {ch: {} for ch in channels}

    for icrate in range(len(vme_configs)):
        vme_config = vme_configs[icrate]
        crate_id = int(vme_config["metadata"]["host"].split("nps-vme")[-1])

        slot_keys = [
            key for key in vme_config.keys() if key.startswith("FADC250_SLOT_")
        ]
        for slot_key in slot_keys:
            slot_id = int(slot_key.split("_")[-1])

            if slot_id not in unique_slots:
                logger.warning(
                    f"Slot {slot_id} in crate {crate_id} not found in channel map, skipping..."
                )
                continue

            mask = (channel_map["crate"] == crate_id) & (channel_map["slot"] == slot_id)
            slot_channels = channel_map.loc[mask]["index"].tolist()

            for k, v in vme_config[slot_key].items():
                if isinstance(v, int) or isinstance(v, float):
                    for ch in slot_channels:
                        vme_config_dict[ch][k] = v
                elif isinstance(v, list):
                    assert len(v) >= len(
                        slot_channels
                    ), f"Not enough values for {k} in crate {crate_id}, slot {slot_id}."
                    for iv, ch in enumerate(slot_channels):
                        vme_config_dict[ch][k] = v[iv]
                else:
                    raise ValueError(
                        f"Unexpected value type for {k} in crate {crate_id}, slot {slot_id}: {type(v)}"
                    )

    df_vme = pd.DataFrame.from_dict(vme_config_dict, orient="index")
    df_vme.to_csv(
        args.output_dir / f"nps_run_{args.run}_vme_config.csv", index_label="channel"
    )
    logger.info(
        f"Saved VME configuration to {args.output_dir / f'nps_run_{args.run}_vme_config.csv'}"
    )

    for icrate in range(len(vtp_configs)):
        vtp_config = vtp_configs[icrate]
        crate_id = int(vtp_config["metadata"]["host"].split("nps-vtp")[-1])

        mask = channel_map["crate"] == crate_id
        slots = channel_map.loc[mask]["slot"].unique().tolist()

        for slot in slots:
            slot_mask = mask & (channel_map["slot"] == slot)
            slot_channels = channel_map.loc[slot_mask]["index"].tolist()
            required_keys = [k for k in vtp_config.keys() if k.startswith("VTP")]

            for k in required_keys:
                v = vtp_config[k]

                if isinstance(v, int) or isinstance(v, float):
                    for ch in slot_channels:
                        vtp_config_dict[ch][k] = v
                elif isinstance(v, list):
                    assert len(v) >= len(
                        slot_channels
                    ), f"Not enough values for {k} in crate {crate_id}."
                    for iv, ch in enumerate(slot_channels):
                        vtp_config_dict[ch][k] = v[iv]
                else:
                    raise ValueError(
                        f"Unexpected value type for {k} in crate {crate_id}: {type(v)}"
                    )

    df_vtp = pd.DataFrame.from_dict(vtp_config_dict, orient="index")
    df_vtp.to_csv(
        args.output_dir / f"nps_run_{args.run}_vtp_config.csv", index_label="channel"
    )
    logger.info(
        f"Saved VME configuration to {args.output_dir / f'nps_run_{args.run}_vme_config.csv'}"
    )


def fetch_nps_config(url: str) -> str:
    """
    Fetch NPS configuration from a given URL.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        code = e.response.status_code
        msg = f"HTTP {code}: {e.response.reason} while fetching {url}"
        if code == 401:
            raise PermissionError(f"Unauthorized — use JLab network.\n{msg}") from e
        elif code == 404:
            raise FileNotFoundError(f"Not found — invalid book or title.\n{msg}") from e
        else:
            raise RuntimeError(f"Unexpected HTTP error.\n{msg}") from e
    except requests.exceptions.Timeout as e:
        raise TimeoutError(f"Request timed out for {url}") from e

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error while fetching {url}: {e}") from e


def to_number(s: str) -> Union[int, float]:
    """
    Convert a string to an int or float.
    """
    try:
        if "." in s or "e" in s or "E" in s:
            return float(s)
        else:
            return int(s)
    except ValueError:
        raise ValueError(f"Unable to convert string to number: {s}")


def parse_line(line: str) -> tuple[str | Union[list[float], float]]:
    """
    Parse a line of configuration.
    """
    content = line.strip().split()
    key = content[0]
    values = content[1:]

    try:
        if len(values) == 1:
            value = to_number(values[0])
        else:
            value = [to_number(v) for v in values]
    except ValueError:
        raise ValueError(f"Unable to convert values to float in line: {line}")

    return key, value


def parse_vtp_config(url: str) -> dict:
    """
    Parse VTP configuration from a given URL.
    Currently a placeholder function.
    """
    text = fetch_nps_config(url)
    result = {"url": url, "metadata": {}}
    try:
        content = [l for l in text.splitlines() if l.strip()]
        metadata = [l.lstrip("#").strip() for l in content if l.startswith("#")]
        slow_controls = [l.strip() for l in content if not l.startswith("#")]

        for l in metadata:
            key, value = l.split(":")
            result["metadata"][key] = value.strip()
        for l in slow_controls:
            key, value = parse_line(l)

            if isinstance(value, list):
                if len(value) == 2:
                    # handles e.g. VTP_NPS_TRIG_DELAY 0 0
                    trgbit = value[0]
                    result[f"{key}_{trgbit}"] = value[1]
                if len(value) == 4:
                    # handles VTP_FIBER_EN 0 1 0 1
                    for iv in range(len(value)):
                        result[f"{key}_{iv}"] = value[iv]
            else:
                result[key] = value

    except Exception as e:
        raise ValueError(f"Error parsing VTP configuration from {url}: {e}") from e

    return result


def parse_vme_config(url: str) -> dict:
    """
    Parse VME configuration from a given URL.
    Currently a placeholder function.
    """
    text = fetch_nps_config(url)
    result = {"url": url, "metadata": {}}
    try:
        content = [l for l in text.splitlines() if l.strip()]
        metadata = [l.lstrip("#").strip() for l in content if l.startswith("#")]
        slow_controls = [l.strip() for l in content if not l.startswith("#")]
        for l in metadata:
            key, value = l.split(":")
            result["metadata"][key] = value.strip()

        slot_line_numbers = [
            i for i, l in enumerate(slow_controls) if l.startswith("FADC250_SLOT")
        ]
        for line_num in slot_line_numbers:
            _, slot = parse_line(slow_controls[line_num])
            result[f"FADC250_SLOT_{slot}"] = {}

            for il in range(line_num + 1, len(slow_controls)):
                if slow_controls[il].startswith("FADC250_SLOT"):
                    break
                key, value = parse_line(slow_controls[il])
                result[f"FADC250_SLOT_{slot}"][key] = value

    except Exception as e:
        raise ValueError(f"Error parsing VME configuration from {url}: {e}") from e

    return result


def fetch_nps_entries(run: int) -> dict:
    """
    Fetch JLog entries for a specific NPS run.
    """
    filters = ["COIN_NPS", f"Start_Run_{run}"]

    try:
        entries = fetch_jlog_entries(
            titles=filters,
            author=None,
            startdate="2023-01-01",
            limit=0,
            books=["HCLOG"],
            fields=["lognumber", "title", "created", "author", "body", "attachments"],
            dryrun=False,
        )
    except FileNotFoundError:
        print(f"No entries found for run {run}.")
        return {}
    except PermissionError as e:
        print(f"Access denied: {e}")
        raise  # re-raise because this is a serious configuration issue
    except Exception as e:
        print(f"Unexpected error fetching run {run}: {e}")
        raise  # re-raise or exit, depending on app design

    data = entries.get("data", None)

    if data is None:
        raise ValueError("Expected 'data' to be absent in JLog entries.")

    assert set(
        [
            "currentItems",
            "totalItems",
            "pageLimit",
            "currentPage",
            "pageCount",
            "entries",
        ]
    ).issubset(data.keys())
    assert data["currentItems"] == 1, f"Expected 1 entry, got {data['currentItems']}"

    lognumber = data["entries"][0]["lognumber"]
    title = data["entries"][0]["title"]
    created = data["entries"][0]["created"]["string"]
    author = data["entries"][0]["author"]
    attachments = data["entries"][0].get("attachments", [])

    if len(attachments) != 14:
        raise ValueError(f"Expected 14 attachment, got {len(attachments)}")

    attachment_urls = [att["url"] for att in attachments]

    return {
        "lognumber": lognumber,
        "title": title,
        "created": created,
        "author": author,
        "attachment_urls": attachment_urls,
    }


def read_json(fname: Union[str, pathlib.Path]):
    fname = pathlib.Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: dict, fname: Union[str, pathlib.Path]):
    fname = pathlib.Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch JLog entries for specific NPS runs."
    )
    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="NPS run number to fetch configuration for.",
    )
    parser.add_argument(
        "--channel-map",
        type=pathlib.Path,
        help="Path to channel map csv file.",
        default="../geo/channel_map.csv",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Enable quiet output."
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Output directory for the fetched configuration.",
        default=".",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
