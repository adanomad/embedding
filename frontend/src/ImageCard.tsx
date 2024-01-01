import React, { useState } from "react";
import { Card, Image, Button, Text, Anchor, Group } from "@mantine/core";

interface ImageCardProps {
  id: string;
  fileName: string;
  flaskEndpoint: string;
  distance?: number;
  onViewAlbum?: (date: string) => void;
  handleFindSimilar?: (id: string) => void;
}

const ImageCard: React.FC<ImageCardProps> = ({
  id,
  fileName,
  flaskEndpoint,
  distance,
  onViewAlbum,
  handleFindSimilar,
}) => {
  const [viewFullSize, setViewFullSize] = useState(false);
  const imageUrl = viewFullSize
    ? `${flaskEndpoint}/image?fileName=${fileName}`
    : `${flaskEndpoint}/thumbnail?fileName=${fileName}`;

  const extractDateFromFileName = (fileName: string): string => {
    return fileName.substring(0, 8); // Assuming 'YYYYMMDD' format
  };

  const formattedDateFromFileName = (fileName: string): string => {
    // Assuming 'YYYYMMDD' format in filename
    const year = fileName.substring(0, 4);
    const month = fileName.substring(4, 6);
    const day = fileName.substring(6, 8);

    const date = new Date(`${year}-${month}-${day}`);
    const formattedDate = `${year}-${month}-${day}`;

    // Calculate days ago
    const today = new Date();
    const differenceInTime = today.getTime() - date.getTime();
    const differenceInDays = Math.floor(differenceInTime / (1000 * 3600 * 24));

    return `${formattedDate} (${differenceInDays} days ago)`;
  };

  const extractedDate = extractDateFromFileName(fileName);
  const formattedDate = formattedDateFromFileName(fileName);

  return (
    <Card shadow="sm" padding="lg">
      <Anchor
        href={`${flaskEndpoint}/image?fileName=${fileName}`}
        target="_blank"
      >
        <Image
          src={imageUrl}
          alt={`View ${fileName}`}
          width={600}
          height={400}
          loading="lazy"
          style={{ cursor: "pointer" }}
        />
      </Anchor>

      <Group style={{ marginTop: "1rem", justifyContent: "space-between" }}>
        <Text>{fileName}</Text>
        <Text size="sm">Similarity Distance: {distance?.toFixed(3)}</Text>
      </Group>

      <Group style={{ marginTop: "0.5rem" }}>
        {handleFindSimilar && (
          <Button
            variant="outline"
            onClick={() => handleFindSimilar(id)}
            size="xs"
          >
            Find Similar Images
          </Button>
        )}

        <Button
          variant="outline"
          onClick={() => setViewFullSize(!viewFullSize)}
          size="xs"
        >
          {viewFullSize ? "Show Thumbnail" : "Show Full Size"}
        </Button>

        {onViewAlbum && (
          <Button
            variant="outline"
            onClick={() => onViewAlbum(extractedDate)}
            size="xs"
          >
            View Images from {formattedDate}
          </Button>
        )}
      </Group>
    </Card>
  );
};

export default ImageCard;
