#!/usr/bin/env python3

import argparse
import requests
import pathlib
from typing import Optional, Union
from collections import OrderedDict
from datetime import datetime


available_fields = [
    "lognumber",
    "title",
    "created",
    "author",
    "entrymakers",
    "numcomments",
    "numattachments",
    "needsattention",
    "tags",
    "body",
    "attachments",
]


def fetch_jlog_entries(
    titles: Optional[Union[list[str], str]] = None,
    author: Optional[str] = None,
    startdate: str = "-180 days",
    enddate: Optional[str] = None,
    limit: int = 0,
    books: Optional[Union[list[str], str]] = None,
    tags: Optional[Union[list[str], str]] = None,
    fields: list[str] = ["lognumber", "title", "body"],
    dryrun: bool = False,
) -> dict:
    """
    Fetch JLab Elog (JLog) entries based on specified filters.
    Parameters
    ----------
    titles : list[str] | str, optional
        Filter entries by title containing these strings.
    author : str, optional
        Filter entries by specific author.
    startdate : str, optional
        Start date for filtering entries (e.g., '-30 days', '2023-01-01'), default is '-180 days'. See http://php.net/manual/en/datetime.formats.php for details.
    enddate : str, optional
        End date for filtering entries (e.g., 'now', '2023-12-31'), default is now().
    limit : int
        Maximum number of entries to return (0 for server max limit), default is 0.
    books : list[str] | str, optional
        Limits entries to a specific logbook.  May be repeated to specify multiple logbooks.  Prefix the string with - (a dash) to inverse the logic and exclude the specified logbook.  Example -TLOG would exclude entries in the Test Logbook.
    tags : list[str] | str, optional
        Limits entries to a specific tag.  May be repeated to specify multiple tags.  Prefix the string with - (a dash) to inverse the logic and exclude the specified tag.  Example -Autolog would exclude entries tagged as Autolog.
    fields : list[str]
        Specifies a field the user would like returned.  The default set of fields will be returned consisting of lognumber, created, title.

    Returns
    -------
    dict
        JSON response from the JLog API containing the filtered entries.
    """

    assert set(fields).issubset(
        set(available_fields)
    ), f"Some fields are not valid. Choose from {available_fields}."
    base_url = "https://logbooks.jlab.org/api/elog/entries?"

    enddate = enddate if enddate else datetime.now().strftime("%Y-%m-%d %H:%M")
    queries = [
        f"limit={limit}",
        f"startdate={startdate}",
        f"enddate={enddate}",
    ]

    if author:
        queries.append(f"author={author}")

    # multiple query parameters
    books = [books] if isinstance(books, str) else books
    tags = [tags] if isinstance(tags, str) else tags
    titles = [titles] if isinstance(titles, str) else titles

    if books:
        queries.extend([f"book={book}" for book in books])
    if tags:
        queries.extend([f"tag={tag}" for tag in tags])
    if titles:
        queries.extend([f"title={title}" for title in titles])
    if fields:
        queries.extend([f"field={field}" for field in fields])

    url = base_url + "&".join(queries)

    if dryrun:
        print(f"Constructed URL: {url}")
        return {}

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        msg = f"HTTP {code}: {e.response.reason} while fetching {url}"
        if code == 400:
            raise ValueError(f"Bad request — check parameters.\n{msg}") from e
        elif code == 401:
            raise PermissionError(f"Unauthorized — use JLab network.\n{msg}") from e
        elif code == 404:
            raise FileNotFoundError(f"Not found — invalid book or title.\n{msg}") from e
        else:
            raise RuntimeError(f"Unexpected HTTP error.\n{msg}") from e

    except requests.exceptions.Timeout as e:
        raise TimeoutError(f"Request timed out for {url}") from e

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error while fetching {url}: {e}") from e


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
        description="Fetch JLab Elog (JLog) entries based on specified filters."
    )
    parser.add_argument(
        "--titles",
        type=str,
        nargs="*",
        help="Filter entries by title containing these strings.",
    )
    parser.add_argument("--author", type=str, help="Filter entries by specific author.")
    parser.add_argument(
        "--startdate",
        type=str,
        default="-180 days",
        help="Start date for filtering entries.",
    )
    parser.add_argument("--enddate", type=str, help="End date for filtering entries.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of entries to return (0 for server max limit).",
    )
    parser.add_argument(
        "--books", type=str, nargs="*", help="Limits entries to a specific logbook."
    )
    parser.add_argument(
        "--tags", type=str, nargs="*", help="Limits entries to a specific tag."
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="*",
        default=["lognumber", "title", "body"],
        help="Specifies fields to be returned.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="jlog_entries.json",
        help="Output JSON file for the fetched entries.",
    )

    args = parser.parse_args()

    entries = fetch_jlog_entries(
        titles=args.titles,
        author=args.author,
        startdate=args.startdate,
        enddate=args.enddate,
        limit=args.limit,
        books=args.books,
        tags=args.tags,
        fields=args.fields,
    )

    if entries:
        write_json(entries, args.output)
        print(
            f"Fetched {len(entries.get('entries', []))} JLog entries and saved to {args.output}."
        )
    else:
        print("No entries fetched.")
