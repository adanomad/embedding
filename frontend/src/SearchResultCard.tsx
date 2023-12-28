import React from "react";
import { Card, Anchor, Text, Image, Button } from "@mantine/core";

interface SearchResultCardProps {
  fileName: string;
  distance: number;
  flaskEndpoint: string;
  onViewAlbum: (date: string) => void; // Callback to view the album
}

const SearchResultCard: React.FC<SearchResultCardProps> = ({
  fileName,
  distance,
  flaskEndpoint,
  onViewAlbum,
}) => {
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
      <Button variant="subtle" onClick={() => onViewAlbum(extractedDate)}>
        View Images from {extractedDate}
      </Button>{" "}
      <Text>Distance: {distance.toFixed(4)}</Text>
      <Image
        src={`${flaskEndpoint}${fileName}`}
        alt={fileName}
        width={400}
        height={400}
      />
    </Card>
  );
};

export default SearchResultCard;
