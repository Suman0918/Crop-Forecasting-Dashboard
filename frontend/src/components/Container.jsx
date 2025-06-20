// Not required in the code above, but if you want a generic wrapper:
import React from "react";

export default function Container({ children }) {
  return <div className="container">{children}</div>;
}
