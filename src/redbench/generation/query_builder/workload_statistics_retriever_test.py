import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from redbench.generation.helper.workload_statistics_retriever import (
    DatabaseStatisticsRetriever,
)


class TestDatabaseStatisticsRetriever(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_retrieve_varchar_lengths(self):
        sql_schema_text = textwrap.dedent(
            """
            CREATE TABLE "simple_table" (
                "id" integer,
                "name" VARCHAR(32),
                "code" character varying(64)
            );

            CREATE TABLE public."Complex Table" (
                "description" char(5),
                "notes" CHARACTER VARYING(255),
                "other" text
            );

            CREATE TABLE my_schema."MixedCase" (
                "value" varchar ( 10 ),
                "other" character varying ( 12 )
            );
            """
        )

        schema_file = self.tmp_path / "schema.sql"
        schema_file.write_text(sql_schema_text)

        stats_path = self.tmp_path / "stats.json"
        stats_path.write_text(json.dumps({}))

        json_schema_path = self.tmp_path / "schema.json"
        json_schema_path.write_text(json.dumps({}))

        retriever = DatabaseStatisticsRetriever(
            num_universes=1,
            column_statistics_path=str(stats_path),
            json_schema_path=str(json_schema_path),
            sql_schema_path=str(schema_file),
        )

        lengths = retriever.retrieve_varchar_lengths()

        self.assertEqual(
            lengths,
            {
                "simple_table": {"name": 32, "code": 64},
                "Complex Table": {"description": 5, "notes": 255},
                "MixedCase": {"value": 10, "other": 12},
            },
        )
        self.assertIs(retriever.retrieve_varchar_lengths(), lengths)

    def test_retrieve_varchar_lengths_imdb(self):
        sql_schema_text = """
            DROP TABLE IF EXISTS "aka_name_0";
CREATE TABLE "aka_name_0"
(
    id            integer NOT NULL PRIMARY KEY,
    person_id     integer NOT NULL,
    name          character varying,
    imdb_index    character varying(3),
    name_pcode_cf character varying(11),
    name_pcode_nf character varying(11),
    surname_pcode character varying(11),
    md5sum        character varying(65)
);

DROP TABLE IF EXISTS "aka_name_1";
CREATE TABLE "aka_name_1"
(
    id            integer NOT NULL PRIMARY KEY,
    person_id     integer NOT NULL,
    name          character varying,
    imdb_index    character varying(3),
    name_pcode_cf character varying(11),
    name_pcode_nf character varying(11),
    surname_pcode character varying(11),
    md5sum        character varying(65)
);

DROP TABLE IF EXISTS "aka_title_0";
CREATE TABLE "aka_title_0"
(
    id              integer NOT NULL PRIMARY KEY,
    movie_id        integer NOT NULL,
    title           character varying,
    imdb_index      character varying(4),
    kind_id         integer NOT NULL,
    production_year integer,
    phonetic_code   character varying(5),
    episode_of_id   integer,
    season_nr       integer,
    episode_nr      integer,
    note            character varying(72),
    md5sum          character varying(32)
);

DROP TABLE IF EXISTS "aka_title_1";
CREATE TABLE "aka_title_1"
(
    id              integer NOT NULL PRIMARY KEY,
    movie_id        integer NOT NULL,
    title           character varying,
    imdb_index      character varying(4),
    kind_id         integer NOT NULL,
    production_year integer,
    phonetic_code   character varying(5),
    episode_of_id   integer,
    season_nr       integer,
    episode_nr      integer,
    note            character varying(72),
    md5sum          character varying(32)
);

DROP TABLE IF EXISTS "cast_info_0";
CREATE TABLE "cast_info_0"
(
    id             integer NOT NULL PRIMARY KEY,
    person_id      integer NOT NULL,
    movie_id       integer NOT NULL,
    person_role_id integer,
    note           character varying,
    nr_order       integer,
    role_id        integer NOT NULL
);

DROP TABLE IF EXISTS "cast_info_1";
CREATE TABLE "cast_info_1"
(
    id             integer NOT NULL PRIMARY KEY,
    person_id      integer NOT NULL,
    movie_id       integer NOT NULL,
    person_role_id integer,
    note           character varying,
    nr_order       integer,
    role_id        integer NOT NULL
);

DROP TABLE IF EXISTS "char_name_0";
CREATE TABLE "char_name_0"
(
    id            integer           NOT NULL PRIMARY KEY,
    name          character varying NOT NULL,
    imdb_index    character varying(2),
    imdb_id       integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum        character varying(32)
);

DROP TABLE IF EXISTS "char_name_1";
CREATE TABLE "char_name_1"
(
    id            integer           NOT NULL PRIMARY KEY,
    name          character varying NOT NULL,
    imdb_index    character varying(2),
    imdb_id       integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum        character varying(32)
);

DROP TABLE IF EXISTS "comp_cast_type_0";
CREATE TABLE "comp_cast_type_0"
(
    id   integer               NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

DROP TABLE IF EXISTS "comp_cast_type_1";
CREATE TABLE "comp_cast_type_1"
(
    id   integer               NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

DROP TABLE IF EXISTS "company_name_0";
CREATE TABLE "company_name_0"
(
    id            integer           NOT NULL PRIMARY KEY,
    name          character varying NOT NULL,
    country_code  character varying(6),
    imdb_id       integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum        character varying(32)
);

DROP TABLE IF EXISTS "company_name_1";
CREATE TABLE "company_name_1"
(
    id            integer           NOT NULL PRIMARY KEY,
    name          character varying NOT NULL,
    country_code  character varying(6),
    imdb_id       integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum        character varying(32)
);

DROP TABLE IF EXISTS "company_type_0";
CREATE TABLE "company_type_0"
(
    id   integer NOT NULL PRIMARY KEY,
    kind character varying(32)
);

DROP TABLE IF EXISTS "company_type_1";
CREATE TABLE "company_type_1"
(
    id   integer NOT NULL PRIMARY KEY,
    kind character varying(32)
);

DROP TABLE IF EXISTS "complete_cast_0";
CREATE TABLE "complete_cast_0"
(
    id         integer NOT NULL PRIMARY KEY,
    movie_id   integer,
    subject_id integer NOT NULL,
    status_id  integer NOT NULL
);

DROP TABLE IF EXISTS "complete_cast_1";
CREATE TABLE "complete_cast_1"
(
    id         integer NOT NULL PRIMARY KEY,
    movie_id   integer,
    subject_id integer NOT NULL,
    status_id  integer NOT NULL
);

DROP TABLE IF EXISTS "info_type_0";
CREATE TABLE "info_type_0"
(
    id   integer               NOT NULL PRIMARY KEY,
    info character varying(32) NOT NULL
);

DROP TABLE IF EXISTS "info_type_1";
CREATE TABLE "info_type_1"
(
    id   integer               NOT NULL PRIMARY KEY,
    info character varying(32) NOT NULL
);

DROP TABLE IF EXISTS "keyword_0";
CREATE TABLE "keyword_0"
(
    id            integer           NOT NULL PRIMARY KEY,
    keyword       character varying NOT NULL,
    phonetic_code character varying(5)
);

DROP TABLE IF EXISTS "keyword_1";
CREATE TABLE "keyword_1"
(
    id            integer           NOT NULL PRIMARY KEY,
    keyword       character varying NOT NULL,
    phonetic_code character varying(5)
);

DROP TABLE IF EXISTS "kind_type_0";
CREATE TABLE "kind_type_0"
(
    id   integer NOT NULL PRIMARY KEY,
    kind character varying(15)
);

DROP TABLE IF EXISTS "kind_type_1";
CREATE TABLE "kind_type_1"
(
    id   integer NOT NULL PRIMARY KEY,
    kind character varying(15)
);

DROP TABLE IF EXISTS "link_type_0";
CREATE TABLE "link_type_0"
(
    id   integer               NOT NULL PRIMARY KEY,
    link character varying(32) NOT NULL
);

DROP TABLE IF EXISTS "link_type_1";
CREATE TABLE "link_type_1"
(
    id   integer               NOT NULL PRIMARY KEY,
    link character varying(32) NOT NULL
);

DROP TABLE IF EXISTS "movie_companies_0";
CREATE TABLE "movie_companies_0"
(
    id              integer NOT NULL PRIMARY KEY,
    movie_id        integer NOT NULL,
    company_id      integer NOT NULL,
    company_type_id integer NOT NULL,
    note            character varying
);

DROP TABLE IF EXISTS "movie_companies_1";
CREATE TABLE "movie_companies_1"
(
    id              integer NOT NULL PRIMARY KEY,
    movie_id        integer NOT NULL,
    company_id      integer NOT NULL,
    company_type_id integer NOT NULL,
    note            character varying
);

DROP TABLE IF EXISTS "movie_info_idx_0";
CREATE TABLE "movie_info_idx_0"
(
    id           integer           NOT NULL PRIMARY KEY,
    movie_id     integer           NOT NULL,
    info_type_id integer           NOT NULL,
    info         character varying NOT NULL,
    note         character varying(1)
);

DROP TABLE IF EXISTS "movie_info_idx_1";
CREATE TABLE "movie_info_idx_1"
(
    id           integer           NOT NULL PRIMARY KEY,
    movie_id     integer           NOT NULL,
    info_type_id integer           NOT NULL,
    info         character varying NOT NULL,
    note         character varying(1)
);

DROP TABLE IF EXISTS "movie_keyword_0";
CREATE TABLE "movie_keyword_0"
(
    id         integer NOT NULL PRIMARY KEY,
    movie_id   integer NOT NULL,
    keyword_id integer NOT NULL
);

DROP TABLE IF EXISTS "movie_keyword_1";
CREATE TABLE "movie_keyword_1"
(
    id         integer NOT NULL PRIMARY KEY,
    movie_id   integer NOT NULL,
    keyword_id integer NOT NULL
);

DROP TABLE IF EXISTS "movie_link_0";
CREATE TABLE "movie_link_0"
(
    id              integer NOT NULL PRIMARY KEY,
    movie_id        integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id    integer NOT NULL
);

DROP TABLE IF EXISTS "movie_link_1";
CREATE TABLE "movie_link_1"
(
    id              integer NOT NULL PRIMARY KEY,
    movie_id        integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id    integer NOT NULL
);

DROP TABLE IF EXISTS "name_0";
CREATE TABLE "name_0"
(
    id            integer           NOT NULL PRIMARY KEY,
    name          character varying NOT NULL,
    imdb_index    character varying(9),
    imdb_id       integer,
    gender        character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum        character varying(32)
);

DROP TABLE IF EXISTS "name_1";
CREATE TABLE "name_1"
(
    id            integer           NOT NULL PRIMARY KEY,
    name          character varying NOT NULL,
    imdb_index    character varying(9),
    imdb_id       integer,
    gender        character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum        character varying(32)
);

DROP TABLE IF EXISTS "role_type_0";
CREATE TABLE "role_type_0"
(
    id   integer               NOT NULL PRIMARY KEY,
    role character varying(32) NOT NULL
);

DROP TABLE IF EXISTS "role_type_1";
CREATE TABLE "role_type_1"
(
    id   integer               NOT NULL PRIMARY KEY,
    role character varying(32) NOT NULL
);

DROP TABLE IF EXISTS "title_0";
CREATE TABLE "title_0"
(
    id              integer           NOT NULL PRIMARY KEY,
    title           character varying NOT NULL,
    imdb_index      character varying(5),
    kind_id         integer           NOT NULL,
    production_year integer,
    imdb_id         integer,
    phonetic_code   character varying(5),
    episode_of_id   integer,
    season_nr       integer,
    episode_nr      integer,
    series_years    character varying(49),
    md5sum          character varying(32)
);

DROP TABLE IF EXISTS "title_1";
CREATE TABLE "title_1"
(
    id              integer           NOT NULL PRIMARY KEY,
    title           character varying NOT NULL,
    imdb_index      character varying(5),
    kind_id         integer           NOT NULL,
    production_year integer,
    imdb_id         integer,
    phonetic_code   character varying(5),
    episode_of_id   integer,
    season_nr       integer,
    episode_nr      integer,
    series_years    character varying(49),
    md5sum          character varying(32)
);

DROP TABLE IF EXISTS "movie_info_0";
CREATE TABLE "movie_info_0"
(
    id           integer           NOT NULL PRIMARY KEY,
    movie_id     integer           NOT NULL,
    info_type_id integer           NOT NULL,
    info         character varying NOT NULL,
    note         character varying
);

DROP TABLE IF EXISTS "movie_info_1";
CREATE TABLE "movie_info_1"
(
    id           integer           NOT NULL PRIMARY KEY,
    movie_id     integer           NOT NULL,
    info_type_id integer           NOT NULL,
    info         character varying NOT NULL,
    note         character varying
);

DROP TABLE IF EXISTS "person_info_0";
CREATE TABLE "person_info_0"
(
    id           integer           NOT NULL PRIMARY KEY,
    person_id    integer           NOT NULL,
    info_type_id integer           NOT NULL,
    info         character varying NOT NULL,
    note         character varying
);

DROP TABLE IF EXISTS "person_info_1";
CREATE TABLE "person_info_1"
(
    id           integer           NOT NULL PRIMARY KEY,
    person_id    integer           NOT NULL,
    info_type_id integer           NOT NULL,
    info         character varying NOT NULL,
    note         character varying
);

            """

        schema_file = self.tmp_path / "schema.sql"
        schema_file.write_text(sql_schema_text)

        stats_path = self.tmp_path / "stats.json"
        stats_path.write_text(json.dumps({}))

        json_schema_path = self.tmp_path / "schema.json"
        json_schema_path.write_text(json.dumps({}))

        retriever = DatabaseStatisticsRetriever(
            num_universes=2,
            column_statistics_path=str(stats_path),
            json_schema_path=str(json_schema_path),
            sql_schema_path=str(schema_file),
        )

        lengths = retriever.retrieve_varchar_lengths()
        expected_lengths = {
            "aka_name_0": {
                "imdb_index": 3,
                "md5sum": 65,
                "name": None,
                "name_pcode_cf": 11,
                "name_pcode_nf": 11,
                "surname_pcode": 11,
            },
            "aka_name_1": {
                "imdb_index": 3,
                "md5sum": 65,
                "name": None,
                "name_pcode_cf": 11,
                "name_pcode_nf": 11,
                "surname_pcode": 11,
            },
            "aka_title_0": {
                "imdb_index": 4,
                "md5sum": 32,
                "note": 72,
                "phonetic_code": 5,
                "title": None,
            },
            "aka_title_1": {
                "imdb_index": 4,
                "md5sum": 32,
                "note": 72,
                "phonetic_code": 5,
                "title": None,
            },
            "cast_info_0": {"note": None},
            "cast_info_1": {"note": None},
            "char_name_0": {
                "imdb_index": 2,
                "md5sum": 32,
                "name": None,
                "name_pcode_nf": 5,
                "surname_pcode": 5,
            },
            "char_name_1": {
                "imdb_index": 2,
                "md5sum": 32,
                "name": None,
                "name_pcode_nf": 5,
                "surname_pcode": 5,
            },
            "comp_cast_type_0": {"kind": 32},
            "comp_cast_type_1": {"kind": 32},
            "company_name_0": {
                "country_code": 6,
                "md5sum": 32,
                "name": None,
                "name_pcode_nf": 5,
                "name_pcode_sf": 5,
            },
            "company_name_1": {
                "country_code": 6,
                "md5sum": 32,
                "name": None,
                "name_pcode_nf": 5,
                "name_pcode_sf": 5,
            },
            "company_type_0": {"kind": 32},
            "company_type_1": {"kind": 32},
            "complete_cast_0": {},
            "complete_cast_1": {},
            "info_type_0": {"info": 32},
            "info_type_1": {"info": 32},
            "keyword_0": {"keyword": None, "phonetic_code": 5},
            "keyword_1": {"keyword": None, "phonetic_code": 5},
            "kind_type_0": {"kind": 15},
            "kind_type_1": {"kind": 15},
            "link_type_0": {"link": 32},
            "link_type_1": {"link": 32},
            "movie_companies_0": {"note": None},
            "movie_companies_1": {"note": None},
            "movie_info_idx_0": {"info": None, "note": 1},
            "movie_info_idx_1": {"info": None, "note": 1},
            "movie_keyword_0": {},
            "movie_keyword_1": {},
            "movie_link_0": {},
            "movie_link_1": {},
            "name_0": {
                "gender": 1,
                "imdb_index": 9,
                "md5sum": 32,
                "name": None,
                "name_pcode_cf": 5,
                "name_pcode_nf": 5,
                "surname_pcode": 5,
            },
            "name_1": {
                "gender": 1,
                "imdb_index": 9,
                "md5sum": 32,
                "name": None,
                "name_pcode_cf": 5,
                "name_pcode_nf": 5,
                "surname_pcode": 5,
            },
            "role_type_0": {"role": 32},
            "role_type_1": {"role": 32},
            "title_0": {
                "imdb_index": 5,
                "md5sum": 32,
                "phonetic_code": 5,
                "series_years": 49,
                "title": None,
            },
            "title_1": {
                "imdb_index": 5,
                "md5sum": 32,
                "phonetic_code": 5,
                "series_years": 49,
                "title": None,
            },
            "movie_info_0": {"info": None, "note": None},
            "movie_info_1": {"info": None, "note": None},
            "person_info_0": {"info": None, "note": None},
            "person_info_1": {"info": None, "note": None},
        }

        self.maxDiff = None
        self.assertEqual(len(lengths), len(expected_lengths))
        for table, cols in expected_lengths.items():
            self.assertIn(table, lengths)
            self.assertEqual(lengths[table], cols, f"Mismatch in table {table}")
        self.assertEqual(lengths, expected_lengths)


if __name__ == "__main__":
    unittest.main()
