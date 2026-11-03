# FMS field tables

`FieldTableParser` reads FMS field tables in the legacy format, the `field_table` file that lists
the tracers of a model built on FMS, such as MOM5 in ACCESS-OM2 and ACCESS-ESM1.6, and MOM6, and
says how each is to be treated. It reads what the field manager reads in
`read_field_table_legacy` of
[`field_manager.F90`](https://github.com/NOAA-GFDL/FMS/blob/main/field_manager/field_manager.F90).

```python
from access.config import FieldTableParser

parser = FieldTableParser()
```

## How a field table reads

An entry is a header naming a field type, a model and a field, followed by the field's methods,
one per line, and closed by a `/`. The configuration reads as the
[FMS YAML field table](https://github.com/NOAA-GFDL/FMS/tree/main/field_manager) lays it out:
three levels of keys for the header, and a dict of the methods for each field. Here is part of
the field table of ACCESS-OM2's one-degree ocean:

```text title="field_table"
"prog_tracers","ocean_mod","temp"
horizontal-advection-scheme = mdppm
vertical-advection-scheme = mdppm
restart_file  = ocean_temp_salt.res.nc
ppm_hlimiter = 3
/
"TRACER", "atmos_mod", "sphum"
           "longname",     "specific humidity"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=3.e-6" /
"rayleigh_damp_table","ocean_mod","rayleigh_damp_table"
"rayleigh","Lombok","itable=36,jtable=112,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"
"rayleigh","Ombai","itable=44,jtable=111,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"/
```

```python
from pathlib import Path

config = parser.parse(Path("field_table").read_text())
print(list(config))
print(dict(config["prog_tracers"]["ocean_mod"]["temp"]))
print(dict(config["tracer"]["atmos_mod"]["sphum"]))
```

```text
['prog_tracers', 'tracer', 'rayleigh_damp_table']
{'horizontal-advection-scheme': 'mdppm', 'vertical-advection-scheme': 'mdppm', 'restart_file': 'ocean_temp_salt.res.nc', 'ppm_hlimiter': 3}
{'longname': 'specific humidity', 'units': 'kg/kg', 'profile_type': [{'value': 'fixed', 'surface_value': 3e-06}]}
```

Keys are matched in any case, and read in lower case, as the field manager reads them: the
field type `"TRACER"` is the key `tracer`. A `#` at the start of a line makes it a comment.

A method is written in one of three ways:

- as two strings, a name and a value: `"units", "kg/kg"`;
- as an assignment, bare or in one string: `ppm_hlimiter = 3`, or `"ppm_hlimiter = 3"`;
- as three strings, a name, a method and a control of comma-separated parameters:
  `"profile_type", "fixed", "surface_value=3.e-6"`. Such a method is a *record*.

## Values

The field manager types a value by its spelling, and so does the parser: `3` is an `int`, `1.0`
and `3.e-6` are floats, `.true.` and `.false.` are logicals, and anything else is a string, as
`mdppm` and `"kg/kg"` are. A string in single quotes, `'3'`, is a string whatever it holds. A
value written with commas, `limits = 1, 2`, is a list.

## Records

A method of three strings is a record: a dict holding the method under `"value"` and each of its
parameters beside it. The lines writing the same method make a list of records, in the order
they are written, and a single one is a list of one:

```python
rayleigh = config["rayleigh_damp_table"]["ocean_mod"]["rayleigh_damp_table"]["rayleigh"]
print(len(rayleigh), dict(rayleigh[0]))
```

```text
2 {'value': 'Lombok', 'itable': 36, 'jtable': 112, 'ktable_1': 1, 'ktable_2': 50, 'rayleigh_damp_table': 5400}
```

A record is a `ConfigRecord`, which is read only: changing one of its fields raises `TypeError`.
It is changed by assigning a whole dict to its place in the list, which rewrites only the
parameters that changed. The list itself can be changed as any list can, and a new record is
written as a copy of its neighbour:

```python
rayleigh[1] = {**rayleigh[1], "itable": 45}
rayleigh.append(
    {"value": "Torres", "itable": 62, "jtable": 106, "ktable_1": 1, "ktable_2": 50, "rayleigh_damp_table": 5400}
)
print("\n".join(str(config).splitlines()[-3:]))
```

```text
"rayleigh","Lombok","itable=36,jtable=112,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"
"rayleigh","Ombai","itable=45,jtable=111,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"
"rayleigh","Torres","itable=62,jtable=106,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"/
```

A list of records keeps at least one: removing the last raises `ValueError`. Delete the method
instead.

## Adding fields and methods

A new method is written like the method above it, and a new field as a new entry, after the last
entry of its type and model, spelling those the way that entry does:

```python
ocean = config["prog_tracers"]["ocean_mod"]
ocean["temp"]["ppm_vlimiter"] = 3
ocean["salt"] = {"restart_file": "ocean_temp_salt.res.nc", "ppm_hlimiter": 3}
print("\n".join(str(config).splitlines()[:11]))
```

```text
"prog_tracers","ocean_mod","temp"
horizontal-advection-scheme = mdppm
vertical-advection-scheme = mdppm
restart_file  = ocean_temp_salt.res.nc
ppm_hlimiter = 3
ppm_vlimiter = 3
/
"prog_tracers","ocean_mod","salt"
restart_file = ocean_temp_salt.res.nc
ppm_hlimiter = 3
/
```

## What is not read

What the field manager would misread or refuse is refused here too, and a file holding any of it
does not parse:

- a method of three strings with no `=` in its control;
- a value holding a `,` or an `=`, or an empty one;
- a comment that does not start its line, or one ending with `/` inside an entry;
- text after the closing `/`;
- a header of fewer than three strings, or of three not parted by commas;
- an entry that is never closed.

The names of a record's parameters keep their case, where the field manager reads them in lower
case. An entry whose header the file writes twice reads as one, as the field manager adds both
to the same list, but no key can be added to it.

Parsing is slower than for the other formats: about half a second for a field table under 10 KB,
and three seconds for the 40 KB tables of the UFS air quality model.
