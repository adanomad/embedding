import React, { useState } from "react";
import { Card, Image, Button, Text, Anchor, Group } from "@mantine/core";

interface ImageCardProps {
  id: string;
  fileName: string;
  distance?: number;
  flaskEndpoint: string;
  onViewAlbum: (date: string) => void;
  handleFindSimilar: (id: string) => void;
}

const ImageCard: React.FC<ImageCardProps> = ({
  id,
  fileName,
  distance,
  flaskEndpoint,
  onViewAlbum,
  handleFindSimilar,
}) => {
  const [viewFullSize, setViewFullSize] = useState(false);
  const imageUrl = viewFullSize
    ? `${flaskEndpoint}/image?fileName=${fileName}`
    : `${flaskEndpoint}/thumbnail?fileName=${fileName}`;

  // Function to extract date from fileName
  const extractDateFromFileName = (fileName: string): string => {
    return fileName.substring(0, 8); // Assuming 'YYYYMMDD' format
  };

  const extractedDate = extractDateFromFileName(fileName);

  return (
    <Card>
      <Image
        src={imageUrl}
        alt={fileName}
        width={600}
        height={600}
        loading="lazy"
      />

      <Group style={{ marginTop: "1rem", flexWrap: "wrap" }}>
        <Anchor href={`${flaskEndpoint}${fileName}`} style={{ flexGrow: 1 }}>
          {fileName}
        </Anchor>

        <Text style={{ flexGrow: 1 }}>{distance?.toFixed(3)}</Text>

        <Button
          variant="subtle"
          onClick={() => handleFindSimilar(id)}
          style={{ flexGrow: 1 }}
        >
          Similar
        </Button>

        <Button
          variant="subtle"
          onClick={() => setViewFullSize(!viewFullSize)}
          style={{ flexGrow: 1 }}
        >
          {viewFullSize ? "Thumbnail" : "Full"}
        </Button>

        {onViewAlbum && (
          <Button
            variant="subtle"
            onClick={() => onViewAlbum(extractedDate)}
            style={{ flexGrow: 1 }}
          >
            Date {extractedDate}
          </Button>
        )}
      </Group>
    </Card>
  );
};

export default ImageCard;
