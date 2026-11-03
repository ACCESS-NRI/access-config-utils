# FMS field tables

`FieldTableParser` reads FMS field tables in the legacy format. This is the `field_table` file of a
model built on FMS, such as MOM6, or MOM5 in ACCESS-OM2 and ACCESS-ESM1.6. It lists the tracers of
the model and says how each one is to be treated. The parser reads what the field manager reads in
`read_field_table_legacy` of
[`field_manager.F90`](https://github.com/NOAA-GFDL/FMS/blob/main/field_manager/field_manager.F90).
You create it like this:

```python
from access.config import FieldTableParser

parser = FieldTableParser()
```

## How a field table reads

An entry starts with a header, which names a field type, a model and a field. The header is followed
by the methods of the field, one on each line, and the entry is closed by a `/`. The configuration
has the same structure as the [FMS YAML field
table](https://github.com/NOAA-GFDL/FMS/tree/main/field_manager): three levels of keys for the
header, and a dict of the methods for each field. As an example, we use part of the field table of
the one-degree ocean of ACCESS-OM2:

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

Save it as `field_table` in the current directory, or adapt the path to where your file is. We parse
it and print the field types, and then the methods of two fields:

```python
from pathlib import Path

config = parser.parse(Path("field_table").read_text())
print(list(config))
print(dict(config["prog_tracers"]["ocean_mod"]["temp"]))
print(dict(config["tracer"]["atmos_mod"]["sphum"]))
```

This prints:

```text
['prog_tracers', 'tracer', 'rayleigh_damp_table']
{'horizontal-advection-scheme': 'mdppm', 'vertical-advection-scheme': 'mdppm', 'restart_file': 'ocean_temp_salt.res.nc', 'ppm_hlimiter': 3}
{'longname': 'specific humidity', 'units': 'kg/kg', 'profile_type': [{'value': 'fixed', 'surface_value': 3e-06}]}
```

Keys are matched in any case, and read in lower case, as the field manager reads them. The field
type `"TRACER"` is therefore the key `tracer`. A `#` at the start of a line makes it a comment.

A method is written in one of three ways:

- as two strings, a name and a value: `"units", "kg/kg"`;
- as an assignment, bare or in one string: `ppm_hlimiter = 3`, or `"ppm_hlimiter = 3"`;
- as three strings, a name, a method and a control of comma-separated parameters:
  `"profile_type", "fixed", "surface_value=3.e-6"`. Such a method is a *record*.

## Values

The field manager decides the type of a value from the way it is written, and so does the parser.
`3` is an `int`, `1.0` and `3.e-6` are floats, and `.true.` and `.false.` are logicals. Anything
else is a string, as `mdppm` and `"kg/kg"` are. A string in single quotes, such as `'3'`, is a
string whatever it contains. A value written with commas, such as `limits = 1, 2`, is a list.

## Records

A method of three strings is a record: a dict that holds the method under `"value"`, and each of its
parameters beside it. The lines that write the same method make a list of records, in the order they
are written. A single line makes a list of one record. For example, the `rayleigh` method of our
field table has two records:

```python
rayleigh = config["rayleigh_damp_table"]["ocean_mod"]["rayleigh_damp_table"]["rayleigh"]
print(len(rayleigh), dict(rayleigh[0]))
```

This prints the number of records and the first one:

```text
2 {'value': 'Lombok', 'itable': 36, 'jtable': 112, 'ktable_1': 1, 'ktable_2': 50, 'rayleigh_damp_table': 5400}
```

A record is a `ConfigRecord`, which is read only: changing one of its fields raises `TypeError`. To
change a record, assign a whole dict to its place in the list. Only the parameters that changed are
rewritten. You can change the list itself like any other list, and a new record is written as a copy
of its neighbour. Here we change one parameter of the second record and append a third one:

```python
rayleigh[1] = {**rayleigh[1], "itable": 45}
rayleigh.append(
    {"value": "Torres", "itable": 62, "jtable": 106, "ktable_1": 1, "ktable_2": 50, "rayleigh_damp_table": 5400}
)
print("\n".join(str(config).splitlines()[-3:]))
```

This prints the last three lines of the file:

```text
"rayleigh","Lombok","itable=36,jtable=112,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"
"rayleigh","Ombai","itable=45,jtable=111,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"
"rayleigh","Torres","itable=62,jtable=106,ktable_1=1,ktable_2=50,rayleigh_damp_table=5400"/
```

A list of records keeps at least one record: removing the last one raises `ValueError`. Delete the
method instead.

## Adding fields and methods

A new method is written like the method above it. A new field is written as a new entry, after the
last entry of its type and model, and its header spells those two as that entry does. Here we add a
method to `temp`, and a new field:

```python
ocean = config["prog_tracers"]["ocean_mod"]
ocean["temp"]["ppm_vlimiter"] = 3
ocean["salt"] = {"restart_file": "ocean_temp_salt.res.nc", "ppm_hlimiter": 3}
print("\n".join(str(config).splitlines()[:11]))
```

This prints the first eleven lines of the file:

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

Anything that the field manager would misread or refuse is refused here too. A file that contains
any of these does not parse:

- a method of three strings with no `=` in its control;
- a value holding a `,` or an `=`, or an empty one;
- a comment that does not start its line, or one ending with `/` inside an entry;
- text after the closing `/`;
- a header of fewer than three strings, or of three not parted by commas;
- an entry that is never closed.

The names of the parameters of a record keep their case, whereas the field manager reads them in
lower case. An entry whose header is written twice in the file is read as one entry, because the
field manager adds both to the same list. No key can be added to such an entry.

Parsing is slower than for the other formats. It takes about half a second for a field table under
10 KB, and three seconds for the 40 KB tables of the UFS air quality model.
