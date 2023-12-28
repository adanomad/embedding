// ImageCard.tsx
// Component for displaying an image card with a thumbnail and a button to view the full size image.
import React, { useState } from "react";
import { Card, Image, Button, Text } from "@mantine/core";

const ImageCard = ({ flaskEndpoint, fileName }) => {
  const [viewFullSize, setViewFullSize] = useState(false);

  const imageUrl = viewFullSize
    ? `${flaskEndpoint}/image?fileName=${fileName}`
    : `${flaskEndpoint}/thumbnail?fileName=${fileName}`;

  return (
    <Card>
      <Text>{fileName}</Text>

      <Image
        src={imageUrl}
        alt={fileName}
        width={600}
        height={600}
        loading="lazy"
      />
      <Button onClick={() => setViewFullSize(true)}>View Full Size</Button>
    </Card>
  );
};

export default ImageCard;
