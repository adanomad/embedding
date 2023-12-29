import React, { useState } from "react";
import { Card, Image, Button, Text, Anchor } from "@mantine/core";

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
      <Anchor href={`${flaskEndpoint}${fileName}`}>
        File Name: {fileName}
      </Anchor>
      <Text>Distance: {distance?.toFixed(4)}</Text>
      <Image
        src={imageUrl}
        alt={fileName}
        width={600}
        height={600}
        loading="lazy"
      />
      <Button variant="subtle" onClick={() => setViewFullSize(!viewFullSize)}>
        {viewFullSize ? "View Thumbnail" : "View Full Size"}
      </Button>
      {onViewAlbum && (
        <Button variant="subtle" onClick={() => onViewAlbum(extractedDate)}>
          View Images from {extractedDate}
        </Button>
      )}
      <Button variant="subtle" onClick={() => handleFindSimilar(id)}>
        Find Similar Images
      </Button>
    </Card>
  );
};

export default ImageCard;
