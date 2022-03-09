from collections import OrderedDict


class DisplayDict(OrderedDict):
    """
    ### Description
    >dict class extension that includes repr_html for key,value display in Jupyter.
    """

    def __init__(self, *args, **kwargs):
        super(DisplayDict, self).__init__(*args, **kwargs)
        self.update(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError

    def __setattr__(self, name, value):
        self[name] = value

    def _repr_html_(self):
        table = [
            "<table>",
            "  <thead>",
            "    <tr><td> </td><th>Key</th><th>Value</th></tr>",
            "  </thead>",
            "  <tbody>",
        ]
        for key, value in self.items():
            table.append(f"<tr><th> {key}</th><th>{value}</th></tr>")
        table.append("</tbody></table>")
        return "\n".join(table)


    def _repr_console_(self):
        """
        ### Description
        Takes own data and organizes it into a console-friendly table.
        """
        final_data = ''
        for key, value in self.items():
            final_data = final_data + str(key) + ":\t"
            final_data = final_data + str(value) + "\t\t|\t\t"
        return final_data

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class OneList(list):
    """
    ### Description:
    >List extension that will return the sole item of the list if len(list) == 1

    ### Usage:
    ```python
    data = {"key":1}
    l = OneList([data])
    assert l == data
    print(l['key'])  #prints 1
    l.append(2)
    print(l[1]) #prints 2
    assert l == data #raises Error
    ```
    """
    def __init__(self, iterable, **kwargs):
        self.l = list(iterable)
        for i in range(0, len(self.l)):
            self.__setitem__(i, self.l[i])
        if hasattr(self, 'value'):
            class_name = type(self.value)
            dr = dir(class_name)
            functions_list = [f for f in dr if not "__" in f]
            for func_name in functions_list:
                setattr(self, func_name, getattr(self.value, func_name))

    def __getattr__(self, name):
        if len(self.l) == 1:
            if hasattr(self.value, name):
                return self.value[name]
            elif hasattr(self, name):
                return self[name]
        else:
            result = self.l.__getattribute__(name)
            return result

    def __getitem__(self, item):
        if len(self.l) == 1:
            if hasattr(self.value, item):
                return self.value[item]
            else:
                return self.value
        else:
            result = self.l.__getitem__(item)
            return result

    def __setitem__(self, key, value):
        if len(self.l) == 1:
            self.value = value
        else:
            self.l.__setitem__(key, value)

    def __add__(self, value):
        return self.l.__add__(value)

    def __contains__(self, key):
        return self.l.__contains__(key)

    def __delitem__(self, key):
        del self.l[key]

    def __eq__(self, value):
        return self.l.__eq__(value)

    def __ge__(self, value):
        return self.l.__ge__(value)

    def __gt__(self, value):
        return self.l.__gt__(value)

    def __iadd__(self, value):
        self.l.__iadd__(value)

    def __imul__(self, value):
        self.l.__imul__(value)

    def __iter__(self):
        return self.l.__iter__()

    def __le__(self, value):
        return self.l.__le__(value)

    def __len__(self):
        return len(self.l)

    def __lt__(self, value):
        return self.__lt__(value)

    def __mul__(self, value):
        return self.l.__mul__(value)

    def __ne__(self, value):
        return self.l.__ne__(value)

    def __repr__(self):
        if len(self.l) == 1:
            return str(self.value)
        else:
            return self.l.__repr__()

    def __reversed__(self):
        return self.l.__reversed__()

    def __rmul__(self, value):
        return self.l.__rmul__(value)

    def __sizeof__(self):
        return self.l.__sizeof__()

    def append(self, obj):
        self.l.append(obj)

    def clear(self):
        self.l.clear()

    def copy(self):
        return self.l.copy()

    def count(self, value):
        return self.l.count(value)

    def extend(self, iterable):
        self.l.extend(iterable)

    def index(self, value, start=0, stop=9223372036854775807):
        return self.l.index(value, start, stop)

    def insert(self, index, obj):
        self.l.insert(index, obj)

    def pop(self, index=-1):
        self.l.pop(index)

    def remove(self, value):
        self.l.remove(value)

    def reverse(self):
        self.l.reverse()

    def sort(self, key=None, reverse=False):
        self.l.sort(key=key, reverse=reverse)